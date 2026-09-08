from matplotlib.colors import LinearSegmentedColormap
from trajectory_prediction import TrajectoryPredictorBase
from typing import Optional
import torch
import os
import numpy as np
import time
import io
import base64
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq
from .data_utils import goal_2d_to_text, decode_trajectory_string
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from peft import PeftModel, PeftConfig
import cv2

from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager


import rospy
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, TransformException
import tf2_ros

from tf_conversions import transformations as tr

def tf_to_matrix(tf_stamped):
    """Convert a geometry_msgs/TransformStamped → 4×4 numpy matrix."""
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    T = tr.translation_matrix((t.x, t.y, t.z))
    R = tr.quaternion_matrix((q.x, q.y, q.z, q.w))
    return T @ R

def create_visualization(image, pred_trajectories, selected_pred_trajectory=None, click_coords=None):
    """Create visualization with overlaid predicted trajectories, a selected trajectory, and click point"""
    # Convert PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    img = image.copy().convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')  # Use RGBA mode for transparency
    
    # Get colors from a colormap for different trajectories
    num_trajectories = len(pred_trajectories)
    
    # Generate colors for trajectories - use HSV color space for better distribution
    def get_colors(n):
        # Use HSV color space to generate evenly distributed colors
        if n <= 10:
            # Use tab10 for small number of trajectories (more distinct)
            cmap = plt.cm.get_cmap('tab10', 10)
            # Avoid red if we use it for the selected trajectory
            colors_rgb = [cmap(i)[:3] for i in range(10)]
            # Example: Skip red (index 3 in tab10 often is red-ish)
            # You might need to adjust this based on the exact colormap
            # colors_rgb = [c for i, c in enumerate(colors_rgb) if i != 3] 
            return colors_rgb[:n] # Return only needed number
        else:
            # Generate colors with good spacing in HSV space
            colors = []
            for i in range(n):
                # Distribute hues evenly, keep saturation and value high for visibility
                h = i / n
                s = 0.7 + (i % 3) * 0.1  # Slight variation in saturation
                v = 0.9
                # Convert HSV to RGB
                r, g, b = plt.cm.hsv(h)[:3]
                # Avoid pure red if used for selected trajectory
                if abs(h - 0.0) < 0.05 and s > 0.8 and v > 0.8: # Check if color is close to red
                    h = (h + 0.1) % 1.0 # Shift hue slightly
                    r, g, b = plt.cm.hsv(h)[:3]
                colors.append((r, g, b))
            return colors
    
    # Get distinct colors for all candidate trajectories
    trajectory_colors = get_colors(num_trajectories)

    # Get the bottom center point of the image for ego motion visualization
    bottom_center = (img.width // 2, img.height - 1)

    # --- Draw candidate trajectories (lower alpha) ---
    candidate_fill_alpha = 80   # Lower alpha for fill
    candidate_line_alpha = 120  # Lower alpha for line
    for traj_idx, trajectory in enumerate(pred_trajectories):
        if trajectory is None or len(trajectory) == 0:
            continue
            
        # Get color for this trajectory
        base_color = tuple(int(x*255) for x in trajectory_colors[traj_idx])
        fill_color = base_color + (candidate_fill_alpha,)
        line_color = base_color + (candidate_line_alpha,)

        num_points = len(trajectory)
        
        for i, (x, y) in enumerate(trajectory):
            radius = 5 # Slightly smaller radius for candidates
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=fill_color,
                outline=line_color, 
                width=1 # Thinner outline
            )
            
            # Connect points with thinner lines
            if i > 0:
                prev_x, prev_y = trajectory[i-1]
                draw.line([(prev_x, prev_y), (x, y)], fill=line_color, width=2) # Thinner line

    # --- Draw selected trajectory (distinct color, higher alpha) ---
    if selected_pred_trajectory is not None and len(selected_pred_trajectory) > 0:
        selected_color_rgb = (0, 255, 0) # Bright Green
        selected_fill_alpha = 180
        selected_line_alpha = 255
        
        fill_color = selected_color_rgb + (selected_fill_alpha,)
        line_color = selected_color_rgb + (selected_line_alpha,)
        
        num_points = len(selected_pred_trajectory)

        # Draw ego motion line from bottom center to first waypoint of selected trajectory
        if num_points > 0:
            first_point = selected_pred_trajectory[0]
            draw.line([bottom_center, (first_point[0], first_point[1])], fill=line_color, width=4) # Thicker line

        for i, (x, y) in enumerate(selected_pred_trajectory):
            radius = 7 # Larger radius for selected
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=fill_color,
                outline=line_color,
                width=2 # Thicker outline
            )
            
            # Connect points with thicker lines
            if i > 0:
                prev_x, prev_y = selected_pred_trajectory[i-1]
                draw.line([(prev_x, prev_y), (x, y)], fill=line_color, width=4) # Thicker line

    # Draw click point (goal) as a red 'X' - Draw last to ensure visibility
    if click_coords is not None:
        x, y = click_coords
        cross_size = 10 # Size of the X arms
        cross_thickness = 3
        # Draw two lines for the X
        draw.line([(x - cross_size, y - cross_size), (x + cross_size, y + cross_size)], fill='red', width=cross_thickness)
        draw.line([(x - cross_size, y + cross_size), (x + cross_size, y - cross_size)], fill='red', width=cross_thickness)

    return img

def traversability_estimator(origin, goals, trav_map, resolution, n=20):
    """Estimate traversability along a path defined by a list of goals."""
    # Get x, y components of the goals
    goals = goals[:, :, :2]  # ignore z
    # Convert meters → cell offsets
    
    goals[:,:,0] /= resolution      # +x → +columns
    goals[:,:,1] /= -resolution      # +y → -rows (image row 0 is top)

    starts = np.zeros_like(goals)
    starts[:,1:,:] = goals[:,:-1,:]

    num_goals = goals.shape[0]

    rows,cols = trav_map.shape

    hypot = np.linalg.norm(goals - starts, axis=2) + 1e-6
    samples = np.linspace(0, 1, n)
    rads = hypot[:, :, np.newaxis] * samples[np.newaxis, np.newaxis, :]
    x_scale = (goals[:, :, 0]-starts[:, :, 0]) / hypot # N,T,2
    y_scale = (goals[:, :, 1]-starts[:, :, 1]) / hypot
    pts = origin + starts[:,:,np.newaxis,:] + rads[:,:,:, np.newaxis] * np.stack((x_scale, y_scale), axis=-1)[:,:,np.newaxis,:]
    pts = np.round(pts).astype(int)

    # Sample traversability, clipping out-of-bounds
    valid_mask = (0 <= pts[..., 1]) & (pts[..., 1] < trav_map.shape[0]) & (0 <= pts[..., 0]) & (pts[..., 0] < trav_map.shape[1])
    valid_pts = np.where(valid_mask[:,:,:,np.newaxis],
        pts,
        origin
    )
    valid_pts = valid_pts.astype(np.int32)
    values = trav_map[valid_pts[..., 1], valid_pts[..., 0]].max(axis=(1,2)).astype(float)

    upscaled_map = cv2.resize(trav_map, (rows*10, cols*10), interpolation=cv2.INTER_LINEAR).T
    upscaled_map = np.clip(upscaled_map * 255.0, 0, 255).astype(np.uint8)
    upscaled_map = np.flipud(upscaled_map)
    plot_goals = (goals.copy() + origin) * 10
    plot_goals[:,:, [0,1]] = plot_goals[:,:, [1,0]]
    plot_goals[:,:,1] = upscaled_map.shape[0] - plot_goals[:,:,1]

    picked_idx = values.argmin()

    img = create_visualization(
        upscaled_map,
        plot_goals,
        plot_goals[picked_idx],
    )
    
    return values.tolist(), img, picked_idx


class VLMTrajectoryPredictor(TrajectoryPredictorBase):
    def __init__(self, node_handle, tf_buffer: tf2_ros.Buffer, time_threshold: Optional[float] = 10000000):
        super().__init__(node_handle, tf_buffer, time_threshold)

        # Load parameters from ROS config
        model_name_or_path = rospy.get_param('~model_name_or_path', "/home/rll/projects/spot_ws/src/research/vamos/models/paligemma2-3b-pt-224-sft-lora-vamos_10pct_gpt5_mini_cocoqa_localized_narratives_fixed")
        self.max_new_tokens = rospy.get_param('~max_new_tokens', 10)
        self.temperature = rospy.get_param('~temperature', 0.1)
        self.num_samples = rospy.get_param('~num_samples', 50)
        self.top_k = rospy.get_param('~top_k', 0)
        self.num_beams = rospy.get_param('~num_beams', 1)
        self.generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": (self.temperature > 0),
            "num_return_sequences": self.num_samples,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "num_beams": self.num_beams,
            "use_cache": True,
        }
        self.torch_dtype = torch.bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_http_server = rospy.get_param('~use_http_server', False)
        self.api_base_url = rospy.get_param('~http_server_url', "http://localhost:8009")
        
        if "lora" in model_name_or_path.lower():
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load processor from base model
            self.processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            
            # Load base model first
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            # Then load LoRA adapter
            self.model = PeftModel.from_pretrained(self.model, model_name_or_path)
            
            print(f"✅ LoRA model loaded successfully from {model_name_or_path} on {self.device.upper()}!")

            print("Merging LoRA adapter...")
            self.model = self.model.merge_and_unload()
            print("✅ ✅ LoRA adapter merged and unloaded.")
        else:
            # Load regular model
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            print(f"✅ Model loaded successfully from {model_name_or_path} on {self.device.upper()}!")

        print("Applying torch.compile...")
        try:
            # Try different modes like 'default', 'reduce-overhead', 'max-autotune'
            # You might need to compile sub-modules or handle dynamic shapes carefully
            self.model = torch.compile(self.model, mode="max-autotune") # Or another mode
            print("✅ torch.compile applied.")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}. Proceeding without compilation.")

        self.model.eval()

        # Set up debug image publishers
        self.debug = rospy.get_param('~debug', True)
        debug_image_topic = rospy.get_param('~debug_image_topic', "/vamos/image_predicted_trajectories")
        value_map_image_topic = rospy.get_param('~value_map_image_topic', "/vamos/value_map_trajectories")
        predicted_paths_2d_topic = rospy.get_param('~predicted_paths_2d_topic', "/vamos/predicted_paths_2d")
        predicted_paths_3d_topic = rospy.get_param('~predicted_paths_3d_topic', "/vamos/predicted_paths_3d")
        predicted_values_topic = rospy.get_param('~predicted_values_topic', "/vamos/predicted_values")
        
        self.debug_image_publisher = rospy.Publisher(
            debug_image_topic,
            RosImage,
            queue_size=1
        )
        self.value_map_image_publisher = rospy.Publisher(
            value_map_image_topic,
            RosImage,
            queue_size=1
        )

        # ----------------------------------------------------------------
        # Publishers for recording all candidate paths and their values
        self.paths_publisher_2d = rospy.Publisher(
            predicted_paths_2d_topic,
            Float32MultiArray,
            queue_size=1
        )
        self.paths_publisher_3d = rospy.Publisher(
            predicted_paths_3d_topic,
            Float32MultiArray,
            queue_size=1
        )
        self.values_publisher = rospy.Publisher(
            predicted_values_topic,
            Float32MultiArray,
            queue_size=1
        )
        # ----------------------------------------------------------------

    def _transform_points_to_frame(
        self,
        points_3d_source: np.ndarray,
        source_frame_id: str,
        target_frame_id: str,
        stamp: rospy.Time,
        timeout: rospy.Duration = rospy.Duration(0.1)
    ) -> Optional[np.ndarray]:
        """
        Transforms an array of 3D points from a source frame to a target frame.
        Returns None if transformation fails for any point or input is invalid.
        Returns an empty (0,3) array if the input points_3d_source is empty (0,3).
        """
        if points_3d_source is None:
            rospy.logwarn_throttle(5.0, "VLMTrajectoryPredictor: _transform_points_to_frame received None for points.")
            return None
        if not isinstance(points_3d_source, np.ndarray) or points_3d_source.ndim != 2 or points_3d_source.shape[1] != 3:
            shape_info = points_3d_source.shape if isinstance(points_3d_source, np.ndarray) else type(points_3d_source)
            rospy.logwarn_throttle(
                5.0,
                f"VLMTrajectoryPredictor: _transform_points_to_frame received mis-shaped or wrong type for points (shape/type: {shape_info}). Expected (N, 3) numpy.ndarray."
            )
            return None
        
        if points_3d_source.shape[0] == 0: # Input is an empty (0,3) array
            return np.array([], dtype=np.float32).reshape(0,3)

        transformed_points_list = []
        for point_s in points_3d_source:
            point_stamped_source = PointStamped()
            point_stamped_source.header.stamp = stamp
            point_stamped_source.header.frame_id = source_frame_id
            point_stamped_source.point.x = float(point_s[0])
            point_stamped_source.point.y = float(point_s[1])
            point_stamped_source.point.z = float(point_s[2])

            try:
                # self.tf_buffer is inherited from TrajectoryPredictorBase.
                # It must be the buffer associated with an active TFListener.
                transformed_point_stamped = self.tf_buffer.transform(
                    point_stamped_source,
                    target_frame_id,
                    timeout=timeout
                )
                p = transformed_point_stamped.point
                transformed_points_list.append([p.x, p.y, p.z])
            except (LookupException, ConnectivityException, ExtrapolationException, TransformException) as e:
                rospy.logwarn(
                    f"VLMTrajectoryPredictor: TF transform failed for point from '{source_frame_id}' "
                    f"to '{target_frame_id}'. Point: {point_s}, Stamp: {stamp.to_sec()}s. Error: {e}"
                )
                return None # If any point fails, the whole batch transformation fails

        return np.array(transformed_points_list, dtype=np.float32)


    def predict_trajectory(self) -> Optional[np.ndarray]:
        if not (self.current_packet and \
                self.current_packet.goal_2d is not None and \
                self.projector.K is not None and \
                self.projector.img_height > 0 and \
                self.current_packet.camera_optical_to_vision_transform is not None and \
                self.current_packet.depth_image is not None):
            rospy.logwarn_throttle(5.0, "VLMTrajectoryPredictor: predict_trajectory called with missing essential data in current_packet or uninitialized projector.")
            return None

        rospy.logdebug("VLMTrajectoryPredictor: Predicting trajectory.")
        total_start_time = time.time()

        # Preprocess data
        preprocess_start_time = time.time()
        image_data = self.current_packet.image
        goal_2d_coords = self.current_packet.goal_2d

        img_width = image_data.shape[1]
        img_height = image_data.shape[0]

        text_prompt = goal_2d_to_text(goal_2d_coords, img_width, img_height, 1024)
        
        preprocess_duration = time.time() - preprocess_start_time
       
        generated_texts = []
        if self.use_http_server:
            try:
                # Convert image to base64
                pil_image = Image.fromarray(image_data).convert('RGB')
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Prepare API request
                api_data = {
                    'image_base64': image_base64,
                    'text_prompt': text_prompt,
                    'max_tokens': str(self.max_new_tokens),
                    'temperature': str(self.temperature),
                    'top_k': str(self.top_k),
                    'num_beams': str(self.num_beams),
                    'num_samples': str(self.num_samples)
                }
                
                # Make API call
                response = requests.post(f"{self.api_base_url}/predict_json", data=api_data, timeout=30)
                
                if response.status_code != 200:
                    rospy.logerr(f"VLM API request failed with status {response.status_code}: {response.text}")
                    return None
                    
                api_result = response.json()
                
                if not api_result.get('success', False):
                    rospy.logerr(f"VLM API returned error: {api_result.get('error_message', 'Unknown error')}")
                    return None
                    
                generated_texts = api_result.get('generated_texts', [])
                api_trajectories = api_result.get('trajectories', [])
            except Exception as e:
                rospy.logerr(f"VLMTrajectoryPredictor: HTTP API call failed: {e}")
                return None
        else:
            command = ["<image><bos>" + text_prompt]
            image_input = [Image.fromarray(image_data).convert('RGB')]

            tokens = self.processor(
                text=command,
                images=image_input,
                return_tensors="pt",
                padding="longest",
            )
            tokens = tokens.to(self.torch_dtype).to(self.device)

            # Run the model
            inference_start_time = time.time()
            with torch.no_grad():
                output_tokens = self.model.generate(**tokens, **self.generation_kwargs)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            inference_duration = time.time() - inference_start_time
            generated_texts = self.processor.batch_decode(output_tokens, skip_special_tokens=True)

        # Decode the output to 2D
        postprocess_start_time = time.time()
        all_pred_coords_2d_lists = []
        for text_output in generated_texts:
            try:
                pred_coords_tensor = decode_trajectory_string(
                    text_output, img_height, img_width, expected_tokens=self.max_new_tokens
                )
                if pred_coords_tensor.numel() > 0:
                    all_pred_coords_2d_lists.append(pred_coords_tensor.tolist())
            except Exception as e:
                rospy.logwarn(f"VLMTrajectoryPredictor: Error decoding trajectory string: {e} for text: {text_output}")
        
        if not all_pred_coords_2d_lists:
            rospy.logwarn("VLMTrajectoryPredictor: No valid 2D trajectories decoded from VLM.")
            return None

        # --- Core logic for 3D processing and transformation ---
        successful_candidates_data = []
        cam_optical_frame_id = self.current_packet.camera_optical_to_vision_transform.child_frame_id
        vision_frame_id = self.current_packet.camera_optical_to_vision_transform.header.frame_id
        base_link_frame_id = rospy.get_param('~robot_frame', 'base_link')
        common_transform_stamp = self.current_packet.camera_optical_to_vision_transform.header.stamp

        for traj_2d_list_candidate in all_pred_coords_2d_lists:
            try:
                if not traj_2d_list_candidate:
                    continue
                
                traj_2d_np = np.array(traj_2d_list_candidate, dtype=np.float32).reshape(-1, 2)
                if traj_2d_np.shape[0] == 0:
                    continue

                # -----------------------------------------------------
                # # Projection using ground plane
                # # 1. Project each 2D pixel straight down onto z = z_ground in vision frame
                # # 1a) camera_optical → vision 4×4
                T_cam2vision = tf_to_matrix(self.current_packet.camera_optical_to_vision_transform)

                # 1b) fetch the feet_center height in vision frame as our ground z
                feet_center_frame = rospy.get_param('~feet_center_frame', 'feet_center')
                feet_tf = self.tf_buffer.lookup_transform(
                    vision_frame_id,
                    feet_center_frame,
                    common_transform_stamp,
                    rospy.Duration(0.3)
                )
                z_ground = feet_tf.transform.translation.z

                # 1c) vectorized projection of all N pixels
                traj_3d_vision = self.projector.pixels_to_ground(
                    traj_2d_np,
                    T_cam2vision,
                    z_ground
                )
                if traj_3d_vision.size == 0:
                    continue

                # 2. Now only need vision → base_link
                traj_3d_base_link = self._transform_points_to_frame(
                    traj_3d_vision,
                    vision_frame_id,
                    base_link_frame_id,
                    common_transform_stamp
                )
                if traj_3d_base_link is None or traj_3d_base_link.shape[0] == 0:
                    continue
                # -----------------------------------------------------

                successful_candidates_data.append(
                    (traj_2d_list_candidate, traj_3d_base_link, traj_3d_vision)
                )
            except Exception as e:
                # Catch any unexpected error during a single candidate's processing
                rospy.logwarn(f"VLMTrajectoryPredictor: Error processing a candidate trajectory: {e}. Skipping candidate.")
                continue # Move to the next candidate

        postprocess_duration = time.time() - postprocess_start_time
        total_duration = time.time() - total_start_time

        rospy.logdebug(f"VLMTrajectoryPredictor Timing: Preproc: {preprocess_duration*1000:.1f}ms, Inference: {inference_duration*1000:.1f}ms, Postproc (incl. 3D): {postprocess_duration*1000:.1f}ms, Total: {total_duration*1000:.1f}ms")

        if not successful_candidates_data:
            rospy.logwarn("VLMTrajectoryPredictor: No candidates successfully processed into 3D base_link and vision frames.")
            return None

        num_points = rospy.get_param('~num_waypoints_to_complete', 5)
        predicted_traj_3d_base_link = [traj_base[:num_points] for traj_2d, traj_base, tajs_vision in successful_candidates_data]  # Extract only the base_link trajectories

        predicted_traj_3d_base_link = np.array(predicted_traj_3d_base_link, dtype=np.float32)
        x_min = 0.0
        x_max = 8.0
        low_mask = predicted_traj_3d_base_link[:, :, 0] < x_min
        rospy.loginfo(low_mask.shape)
        predicted_traj_3d_base_link[low_mask, 0] = x_max

        h, w = self.current_packet.value_map.shape
        # assume robot is at the center cell
        center = np.array([0.0 , h // 2])

        values, img, picked_idx = traversability_estimator(origin=center, goals=predicted_traj_3d_base_link.copy(), trav_map=self.current_packet.value_map, resolution=0.25)
        # plotting
        extent = [0, 5, -2.5, 2.5]
        points = predicted_traj_3d_base_link.reshape(-1, 3)[:, :2]  # Flatten to (N, 2)
        # Plot the image
        plt.figure()
        im = plt.imshow(self.current_packet.value_map, extent=extent, origin='upper', aspect='auto')
        plt.colorbar(im, label='Traversability Value')
        white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
        # Example points to overlay
        sc = plt.scatter(points[:, 0], points[:, 1], c = np.repeat(values,num_points), cmap=white_red, marker='o')
        plt.colorbar(sc, label='Value')

        # Save the figure
        plt.savefig('/home/rll/Pictures/traversability_plot.png', dpi=300)
        plt.close()

        selected_value_idx = np.argmin(values)

        selected_2d_list_for_viz, _selected_traj_3d_base_link, selected_traj_3d_vision_to_return = successful_candidates_data[selected_value_idx]
        
        # ----------------------------------------------------------------
        # publish all predicted 2D paths (shape: [num_candidates, num_points, 2])
        # publish all 2D candidate trajectories (shape: [num_candidates, num_points, 2])
        traj2d = np.array([traj2d for traj2d, _, _ in successful_candidates_data], dtype=np.float32)
        paths_msg_2d = Float32MultiArray()
        paths_msg_2d.data = traj2d.flatten().tolist()
        self.paths_publisher_2d.publish(paths_msg_2d)

        # publish all predicted 3D paths (shape: [num_candidates, num_points, 3])
        paths_msg_3d = Float32MultiArray()
        paths_msg_3d.data = predicted_traj_3d_base_link.flatten().tolist()
        self.paths_publisher_3d.publish(paths_msg_3d)

        # publish all traversability values (one per candidate)
        values_msg = Float32MultiArray()
        values_msg.data = values
        self.values_publisher.publish(values_msg)
        # ----------------------------------------------------------------

        if self.debug:
            goal_coords_for_viz = tuple(map(int, goal_2d_coords)) if goal_2d_coords is not None else None
            selected_for_viz = selected_2d_list_for_viz
            other_trajs_for_viz = [data[0] for i, data in enumerate(successful_candidates_data) if i != 0]

            try:
                value_img_np = np.array(img)
                ros_image_msg = RosImage()
                ros_image_msg.header.stamp = rospy.Time.now()
                ros_image_msg.header.frame_id = "value_img_frame"
                ros_image_msg.height = value_img_np.shape[0]
                ros_image_msg.width = value_img_np.shape[1]
                
                ros_image_msg.encoding = "rgb8"
                ros_image_msg.step = value_img_np.shape[1] * 3

                ros_image_msg.is_bigendian = 0
                ros_image_msg.data = value_img_np.tobytes()
                self.value_map_image_publisher.publish(ros_image_msg)
            except Exception as e:
                rospy.logerr(f"VLMTrajectoryPredictor: Error publishing debug image: {e}")

            debug_image_pil = create_visualization(
                image_data,
                other_trajs_for_viz,
                selected_pred_trajectory=selected_for_viz,
                click_coords=goal_coords_for_viz
            )
            try:
                debug_image_np = np.array(debug_image_pil)
                ros_image_msg = RosImage()
                ros_image_msg.header.stamp = rospy.Time.now()
                ros_image_msg.header.frame_id = "debug_image_frame"
                ros_image_msg.height = debug_image_np.shape[0]
                ros_image_msg.width = debug_image_np.shape[1]
                
                if debug_image_np.ndim == 3 and debug_image_np.shape[2] == 3:
                    ros_image_msg.encoding = "rgb8"
                    ros_image_msg.step = debug_image_np.shape[1] * 3
                else:
                     rospy.logerr("VLMTrajectoryPredictor: Unsupported image format for debug ROS conversion")
                     return selected_traj_3d_vision_to_return

                ros_image_msg.is_bigendian = 0
                ros_image_msg.data = debug_image_np.tobytes()
                self.debug_image_publisher.publish(ros_image_msg)
            except Exception as e:
                rospy.logerr(f"VLMTrajectoryPredictor: Error publishing debug image: {e}")

        return selected_traj_3d_vision_to_return
    
    def predict_turn_to_goal_trajectory(self) -> Optional[np.ndarray]:
        """
        Calculates a 3D waypoint in the 'vision' frame intended to make the
        robot turn towards the goal.
        It uses a 'behind-and-side' waypoint calculated in camera_optical_frame
        and then transforms it to the vision frame.
        """
        rospy.logdebug("VLMTrajectoryPredictor: Predicting turn-to-goal trajectory.") # Changed from print

        if not (self.current_packet and \
                self.current_packet.camera_optical_to_vision_transform is not None):
            rospy.logwarn_throttle(5.0, "VLMTrajectoryPredictor: predict_turn_to_goal_trajectory missing essential packet data for transformation.")
            return None

        # 1. Calculate waypoint in camera_optical_frame using the base class method
        # super() ensures we call TrajectoryPredictorBase._calculate_behind_and_side_wp_cam()
        wp_3d_cam_optical = super()._calculate_behind_and_side_wp_cam()

        if wp_3d_cam_optical is None or wp_3d_cam_optical.shape[0] == 0:
            rospy.logwarn("VLMTrajectoryPredictor: _calculate_behind_and_side_wp_cam returned None or empty for turning.")
            return None

        # 2. Transform this waypoint to the 'vision' frame
        cam_optical_frame_id = self.current_packet.camera_optical_to_vision_transform.child_frame_id
        vision_frame_id = self.current_packet.camera_optical_to_vision_transform.header.frame_id
        transform_stamp = self.current_packet.camera_optical_to_vision_transform.header.stamp
        
        wp_3d_vision = self._transform_points_to_frame(
            wp_3d_cam_optical,
            cam_optical_frame_id,
            vision_frame_id,
            transform_stamp
        )

        if wp_3d_vision is None or wp_3d_vision.shape[0] == 0:
            rospy.logwarn("VLMTrajectoryPredictor: Failed to transform 'behind-and-side' waypoint to vision frame.")
            return None
        
        # wp_3d_vision should be a (1, 3) numpy array
        return wp_3d_vision
