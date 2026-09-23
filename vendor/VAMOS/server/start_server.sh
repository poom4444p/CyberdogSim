#!/bin/bash

# VLM Navigation API Server Startup Script

# Default values
HOST="0.0.0.0"
PORT="8009"
MODEL_PATH="mateoguaman/vamos"
USE_FP32=""
TRUST_REMOTE_CODE="--trust_remote_code"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --use_fp32)
            USE_FP32="--use_fp32"
            shift
            ;;
        --no_trust_remote_code)
            TRUST_REMOTE_CODE=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST                 Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT                 Port to bind to (default: 8000)"
            echo "  --model_path PATH           Path to model to preload"
            echo "  --use_fp32                  Use FP32 precision"
            echo "  --no_trust_remote_code      Don't trust remote code"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_path /path/to/model"
            echo "  $0 --host 127.0.0.1 --port 8080 --model_path /path/to/model"
            echo "  $0 --model_path /path/to/model --use_fp32"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting VLM Navigation API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
if [ -n "$MODEL_PATH" ]; then
    echo "Model: $MODEL_PATH"
fi
if [ -n "$USE_FP32" ]; then
    echo "Precision: FP32"
else
    echo "Precision: bfloat16"
fi
echo ""

# Build command
CMD="python vlm_server.py --host $HOST --port $PORT"
if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi
if [ -n "$USE_FP32" ]; then
    CMD="$CMD $USE_FP32"
fi
if [ -n "$TRUST_REMOTE_CODE" ]; then
    CMD="$CMD $TRUST_REMOTE_CODE"
fi

echo "Running: $CMD"
echo ""

# Run the server
exec $CMD