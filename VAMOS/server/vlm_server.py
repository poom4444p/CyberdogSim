import os
import argparse
import json
import re
import torch
import torch._inductor
import torch._dynamo
import torch.compiler
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftConfig, PeftModel
import io
import base64
from typing import List, Optional, Dict, Any
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PyTorch compilation features that cause CUDA graph errors
try:
    # Disable CUDA graphs in different PyTorch subsystems
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fallback_random = True  # Use fallbacks for random ops
except AttributeError as e:
    logger.warning(f"Could not configure PyTorch inductor: {e}")

try:
    # Disable CUDNN benchmark
    torch.backends.cudnn.benchmark = False
except AttributeError as e:
    logger.warning(f"Could not configure CUDNN benchmark: {e}")

try:
    # Configure PyTorch dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 64
except AttributeError as e:
    logger.warning(f"Could not configure PyTorch dynamo: {e}")

# Disable torch.compile if available
if hasattr(torch, "compile"):
    torch.compile = lambda *args, **kwargs: args[0]

# Global variables for model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
# Global variable for natural language steering instructions
steering_instructions = ""
return_trajectories = False
latest_image = {"content": None, "content_type": None}
text_coordinates = ""
response = None

# FastAPI app
app = FastAPI(
    title="VLM Navigation API",
    description="Vision Language Model for Navigation Trajectory Prediction",
    version="1.0.0"
)

# Increase file upload size limits
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Set max file size to 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to handle large file uploads
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"}
            )
    response = await call_next(request)
    return response

# Pydantic models for request/response
class NavigationRequest(BaseModel):
    text_prompt: str
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: int = 100
    num_beams: int = 1
    num_samples: int = 1

class NavigationResponse(BaseModel):
    success: bool
    trajectories: List[List[List[float]]]
    generated_texts: List[str]
    error_message: Optional[str] = None
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: Optional[str] = None

class ModelLoadRequest(BaseModel):
    model_path: str
    use_fp32: bool = False
    trust_remote_code: bool = True

class ModelLoadResponse(BaseModel):
    success: bool
    message: str

class SteeringInstructionsRequest(BaseModel):
    instructions: str

class SteeringInstructionsResponse(BaseModel):
    success: bool
    current_instructions: str
    message: str

# Utility functions from the original app
def point_to_token(point):
    """Convert coordinate to token format"""
    x_token = f"<loc{int(point[0]):04d}>"
    y_token = f"<loc{int(point[1]):04d}>"
    return [x_token, y_token]
    
def decode_token_to_coordinates(token):
    """Convert a location token back to x,y coordinates"""
    num = int(token[4:8])
    return num

def decode_trajectory_string(trajectory_string, img_height, img_width, expected_tokens=10):
    """Convert a string of location tokens into a list of coordinates."""
    # Use regex to find all <loc> tokens
    generated_traj = trajectory_string.split("\n")[-1]
    tokens = re.findall(r'<loc\d{4}>', generated_traj)
    
    if len(tokens) < 2:
        return torch.empty((0, 2), dtype=torch.float32)
    
    # Convert tokens to coordinates
    coords = [decode_token_to_coordinates(token) for token in tokens]

    # Padding logic
    if len(coords) < expected_tokens:
        if len(coords) % 2 != 0:
            coords = coords[:-1]
            
        if len(coords) >= 2:
            last_pair = coords[-2:]
            while len(coords) < expected_tokens:
                coords.extend(last_pair)
        elif len(coords) == 0:
             return torch.empty((0, 2), dtype=torch.float32)

    if not coords:
        return torch.empty((0, 2), dtype=torch.float32)

    coords_tensor = torch.tensor(coords).reshape(-1, 2).float()
    
    # Normalize coordinates back to image space
    if img_width > 0:
        coords_tensor[:, 0] = coords_tensor[:, 0] * img_width / 1024
    else:
         coords_tensor[:, 0] = 0
    if img_height > 0:
        coords_tensor[:, 1] = coords_tensor[:, 1] * img_height / 1024
    else:
         coords_tensor[:, 1] = 0
    
    return coords_tensor

def load_model(model_path, use_fp32=False, trust_remote_code=True):
    """Load the VLM model and processor."""
    global model, processor, device
    
    if not model_path or model_path.strip() == "":
        return False, "Error: Please provide a valid model path."
    
    try:
        if "lora" in model_path.lower():
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load processor from base model
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Load base model first
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32 if use_fp32 else torch.bfloat16,
                trust_remote_code=trust_remote_code
            ).to(device)
            
            # Then load LoRA adapter
            model = PeftModel.from_pretrained(model, model_path)
            
            logger.info(f"LoRA model loaded: model={model is not None}, processor={processor is not None}")
            logger.info(f"Model object: {type(model)}, Processor object: {type(processor)}")
            logger.info(f"Global model id: {id(model)}, Global processor id: {id(processor)}")
            return True, f"LoRA model loaded successfully from {model_path} on {device.upper()}!"
        else:
            # Load regular model
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float32 if use_fp32 else torch.bfloat16,
                trust_remote_code=trust_remote_code
            ).to(device)
            
            return True, f"Model loaded successfully from {model_path} on {device.upper()}!"
    except Exception as e:
        # Clear memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return False, f"Error loading model: {str(e)}"

def generate_trajectories(image, text_prompt, max_tokens, temperature, top_k, num_beams, num_samples):
    """Generate trajectories for a given prompt"""
    global model, processor, device, steering_instructions

    # Append steering instructions to the prompt if they exist
    full_prompt = text_prompt
    if steering_instructions.strip():
        full_prompt = f"{text_prompt}.{steering_instructions.strip()}"
    
    logger.info(f"Using prompt: {full_prompt}")
    
    # Prepare model input
    model_input = processor(
        text=["<image><bos>" + full_prompt],
        images=[image.convert("RGB")],
        return_tensors="pt",
        padding="longest",
    ).to(device)
    
    try:
        with torch.no_grad():
            if num_beams > 1 and temperature > 0:
                logger.warning("Using beam search with temperature can cause errors. Setting temperature to 0.")
                temperature = 0
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": (temperature > 0),
                "num_return_sequences": num_samples
            }
            
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            if top_k > 0:
                generation_kwargs["top_k"] = top_k
            if num_beams > 1:
                generation_kwargs["num_beams"] = num_beams
                
            outputs = model.generate(
                **model_input,
                **generation_kwargs
            )
    except Exception as e:
        error_message = f"Error during model generation: {str(e)}"
        logger.error(error_message)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return [], [], error_message
    
    # Decode output sequences
    generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)
    
    all_pred_coords = []
    valid_generated_texts = []
    
    # Process each generated sequence
    for text in generated_texts:
        try:
            pred_coords = decode_trajectory_string(
                text, 
                image.size[1],  # height
                image.size[0],  # width
                expected_tokens=max_tokens
            )
            if pred_coords.numel() > 0:
                all_pred_coords.append(pred_coords.tolist())
                valid_generated_texts.append(text)
        except Exception as e:
            logger.warning(f"Error decoding trajectory: {e} for text: {text}")
    return all_pred_coords, valid_generated_texts, None

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    try:
        model_path = None
        if model is not None:
            try:
                model_path = getattr(model, 'config', {}).get('_name_or_path', None)
            except:
                model_path = "Unknown"
        
        return HealthResponse(
            status="VLM Navigation API is running",
            model_loaded=model is not None and processor is not None,
            device=device,
            model_path=model_path
        )
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return HealthResponse(
            status="VLM Navigation API is running (error getting model info)",
            model_loaded=False,
            device=device,
            model_path=None
        )

@app.get("/health")
def health():
    """Health check endpoint - simplified like NaVILA"""
    return JSONResponse({
        "status": "ok", 
        "cuda": torch.cuda.is_available(), 
        "gpus": torch.cuda.device_count(),
        "model_loaded": model is not None and processor is not None,
        "device": device
    })

@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model_endpoint(request: ModelLoadRequest):
    """Load a VLM model"""
    success, message = load_model(
        request.model_path, 
        request.use_fp32, 
        request.trust_remote_code
    )
    return ModelLoadResponse(success=success, message=message)

@app.post("/predict", response_model=NavigationResponse)
async def predict_navigation(
    image: UploadFile = File(...),
    text_prompt: str = Form(...),
    max_tokens: int = Form(10),
    temperature: float = Form(1.0),
    top_k: int = Form(100),
    num_beams: int = Form(1),
    num_samples: int = Form(1)
):
    """
    Predict navigation trajectories from an image and text prompt.
    
    Args:
        image: Image file (PNG, JPG, etc.)
        text_prompt: Text description of the navigation task
        max_tokens: Maximum tokens to generate (default: 10)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 100)
        num_beams: Number of beams for beam search (default: 1)
        num_samples: Number of samples to generate (default: 1)
    
    Returns:
        NavigationResponse with predicted trajectories and generated text
    """
    global model, processor, device
    import time
    start_time = time.time()
    
    if model is None or processor is None:
        logger.error(f"Model/processor check failed: model={model is not None}, processor={processor is not None}")
        logger.error(f"Model object: {type(model)}, Processor object: {type(processor)}")
        logger.error(f"Global model id: {id(model) if model else 'None'}, Global processor id: {id(processor) if processor else 'None'}")
        processing_time = time.time() - start_time
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message="Model not loaded. Please load a model first using /load_model endpoint.",
            processing_time=processing_time
        )
    
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate trajectories
        trajectories, generated_texts, error = generate_trajectories(
            pil_image, text_prompt, max_tokens, temperature, top_k, num_beams, num_samples
        )
        
        processing_time = time.time() - start_time
        
        if error:
            return NavigationResponse(
                success=False,
                trajectories=[],
                generated_texts=[],
                error_message=error,
                processing_time=processing_time
            )
        
        return NavigationResponse(
            success=True,
            trajectories=trajectories,
            generated_texts=generated_texts,
            error_message=None,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in prediction: {str(e)}")
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message=f"Error processing request: {str(e)}",
            processing_time=processing_time
        )

@app.post("/predict_json", response_model=NavigationResponse)
async def predict_navigation_json(
    image_base64: str = Form(...),
    text_prompt: str = Form(...),
    max_tokens: int = Form(10),
    temperature: float = Form(1.0),
    top_k: int = Form(100),
    num_beams: int = Form(1),
    num_samples: int = Form(1)
):
    """
    Predict navigation trajectories from a base64-encoded image and text prompt.
    
    Args:
        image_base64: Base64-encoded image string
        text_prompt: Text description of the navigation task
        max_tokens: Maximum tokens to generate (default: 10)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 100)
        num_beams: Number of beams for beam search (default: 1)
        num_samples: Number of samples to generate (default: 1)
    
    Returns:
        NavigationResponse with predicted trajectories and generated text
    """
    global model, processor, device, return_trajectories, latest_image, response, text_coordinates
    return_trajectories = False
    import time
    start_time = time.time()
    
    if model is None or processor is None:
        logger.error(f"Model/processor check failed: model={model is not None}, processor={processor is not None}")
        logger.error(f"Model object: {type(model)}, Processor object: {type(processor)}")
        logger.error(f"Global model id: {id(model) if model else 'None'}, Global processor id: {id(processor) if processor else 'None'}")
        processing_time = time.time() - start_time
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message="Model not loaded. Please load a model first using /load_model endpoint.",
            processing_time=processing_time
        )
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        latest_image["content"] = buffer.getvalue()
        latest_image["content_type"] = "image/png"
        text_coordinates = text_prompt
        print(f"Received text prompt: {text_prompt}")
        
        # Generate trajectories
        trajectories, generated_texts, error = generate_trajectories(
            pil_image, text_prompt, max_tokens, temperature, top_k, num_beams, num_samples
        )
        
        processing_time = time.time() - start_time
        
        if error:
            return NavigationResponse(
                success=False,
                trajectories=[],
                generated_texts=[],
                error_message=error,
                processing_time=processing_time
            )
        
        # while not return_trajectories:
        #     await asyncio.sleep(0.2)

        # logger.info(response.generated_texts)
        
        return NavigationResponse(
            success=True,
            trajectories=trajectories,
            generated_texts=generated_texts,
            error_message=None,
            processing_time=processing_time
        )
                
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in prediction: {str(e)}")
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message=f"Error processing request: {str(e)}",
            processing_time=processing_time
        )

@app.post("/predict_client_json", response_model=NavigationResponse)
async def predict_navigation_client_json(
    image_base64: str = Form(...),
    text_prompt: str = Form(...),
    max_tokens: int = Form(10),
    temperature: float = Form(1.0),
    top_k: int = Form(100),
    num_beams: int = Form(1),
    num_samples: int = Form(1)
):
    """
    Predict navigation trajectories from a base64-encoded image and text prompt.
    
    Args:
        image_base64: Base64-encoded image string
        text_prompt: Text description of the navigation task
        max_tokens: Maximum tokens to generate (default: 10)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling parameter (default: 100)
        num_beams: Number of beams for beam search (default: 1)
        num_samples: Number of samples to generate (default: 1)
    
    Returns:
        NavigationResponse with predicted trajectories and generated text
    """
    global model, processor, device, response
    import time
    start_time = time.time()
    
    if model is None or processor is None:
        logger.error(f"Model/processor check failed: model={model is not None}, processor={processor is not None}")
        logger.error(f"Model object: {type(model)}, Processor object: {type(processor)}")
        logger.error(f"Global model id: {id(model) if model else 'None'}, Global processor id: {id(processor) if processor else 'None'}")
        processing_time = time.time() - start_time
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message="Model not loaded. Please load a model first using /load_model endpoint.",
            processing_time=processing_time
        )
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        # print(f"Decoded image data of length {len(image_data)}")
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate trajectories
        trajectories, generated_texts, error = generate_trajectories(
            pil_image, text_prompt, max_tokens, temperature, top_k, num_beams, num_samples
        )
        
        processing_time = time.time() - start_time
        
        if error:
            return NavigationResponse(
                success=False,
                trajectories=[],
                generated_texts=[],
                error_message=error,
                processing_time=processing_time
            )
        
        response = NavigationResponse(
            success=True,
            trajectories=trajectories,
            generated_texts=generated_texts,
            error_message=None,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in prediction: {str(e)}")
        return NavigationResponse(
            success=False,
            trajectories=[],
            generated_texts=[],
            error_message=f"Error processing request: {str(e)}",
            processing_time=processing_time
        )

@app.post("/steering_instructions", response_model=SteeringInstructionsResponse)
async def set_steering_instructions(request: SteeringInstructionsRequest):
    """
    Set natural language steering instructions that will be appended to all navigation prompts.
    
    Args:
        request: SteeringInstructionsRequest containing the instructions
    
    Returns:
        SteeringInstructionsResponse with success status and current instructions
    """
    global steering_instructions
    
    try:
        steering_instructions = request.instructions
        logger.info(f"Updated steering instructions: '{steering_instructions}'")
        
        return SteeringInstructionsResponse(
            success=True,
            current_instructions=steering_instructions,
            message=f"Steering instructions updated successfully. Current: '{steering_instructions}'"
        )
    except Exception as e:
        logger.error(f"Error setting steering instructions: {str(e)}")
        return SteeringInstructionsResponse(
            success=False,
            current_instructions=steering_instructions,
            message=f"Error setting steering instructions: {str(e)}"
        )

@app.get("/steering_instructions", response_model=SteeringInstructionsResponse)
async def get_steering_instructions():
    """
    Get the current natural language steering instructions.
    
    Returns:
        SteeringInstructionsResponse with current instructions
    """
    global steering_instructions
    
    return SteeringInstructionsResponse(
        success=True,
        current_instructions=steering_instructions,
        message=f"Current steering instructions: '{steering_instructions}'"
    )

@app.get("/get_image", response_model=None)
async def get_image():
    """
    Serves a file to the frontend as if it were an UploadFile-like response.
    """
    global latest_image
    if latest_image["content"] is None:
        raise HTTPException(status_code=404, detail="No image available")

    return StreamingResponse(
        io.BytesIO(latest_image["content"]),
        media_type=latest_image["content_type"],
        headers={"Content-Disposition": "inline; filename=image.png"}
    )

@app.post("/vamos")
async def set_vamos():
    global return_trajectories
    return_trajectories = True
    return {"message": "vamos_flag set to True"}

@app.post("/get_goal_coordinates")
async def get_goal_coordinates():
    global text_coordinates
    # Find all numbers after 'loc'
    numbers = re.findall(r'loc(\d+)', text_coordinates)
    
    if len(numbers) < 2:
        return {"error": "Not enough loc numbers found"}
    
    # Convert to integers
    x, y = int(numbers[0]), int(numbers[1])
    
    return {"x": x, "y": y}

@app.get("/")
@app.get("/interface")
async def get_interface():
    """
    Serve the HTML interface for testing VLM navigation with steering instructions.
    """
    html_path = os.path.join(os.path.dirname(__file__), "vlm_navigation_interface.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse(
        status_code=404,
        content={"error": f"Interface file not found at {html_path}"}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Navigation API Server")
    parser.add_argument("--host", type=str, default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--model_path", type=str, required=True, help="Path to preload model")
    parser.add_argument("--use_fp32", action="store_true", help="Use FP32 precision")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Trust remote code")
    
    args = parser.parse_args()
    
    # Use environment variables with fallbacks (like NaVILA server)
    host = args.host or os.environ.get("BIND_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("PORT", "8009"))  # Use port 8009 like NaVILA
    
    # Preload model if specified via command line argument
    if args.model_path:
        logger.info(f"Preloading model from {args.model_path}...")
        success, message = load_model(args.model_path, args.use_fp32, args.trust_remote_code)
        if success:
            logger.info(f"Model loaded successfully: {message}")
        else:
            logger.error(f"Failed to load model: {message}")
            logger.error("Server will start but model endpoints will not work until model is loaded via /load_model endpoint")
    else:
        logger.warning("No model path provided. Server will start but model endpoints will not work until model is loaded via /load_model endpoint")
    
    # Start server with uvicorn.run like NaVILA
    logger.info(f"Starting VLM Navigation server on {host}:{port}")
    # Configure uvicorn with larger limits
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port, 
        workers=1,
        limit_max_requests=1000,
        limit_concurrency=100,
        timeout_keep_alive=30
    )
    server = uvicorn.Server(config)
    server.run()