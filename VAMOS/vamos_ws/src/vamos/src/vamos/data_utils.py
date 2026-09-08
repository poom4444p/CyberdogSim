import torch
from typing import Optional
from transformers import AutoProcessor
import json
import re

# ----------------------------------------------------------------------------
# Utilities
def point_to_token(point):
    """Convert coordinate to token format"""
    x_token = f"<loc{int(point[0]):04d}>"
    y_token = f"<loc{int(point[1]):04d}>"
    return [x_token, y_token]

def goal_2d_to_text(goal_2d, img_width, img_height, grid_size=1024):
    """Convert goal coordinates to text format."""
    x = int(goal_2d[0] / img_width * grid_size)
    y = int(goal_2d[1] / img_height * grid_size)
    text = f"Navigate to x=<loc{x:04d}>, y=<loc{y:04d}>"
    return text


def pad_trajectory(traj, target_length=50):
    """Pad trajectory with last value until it reaches target length."""
    if len(traj) >= target_length:
        return traj
    
    padding_needed = target_length - len(traj)
    last_value = traj[-1]
    return traj + [last_value] * padding_needed

def subsample_trajectory(trajectory, num_points=5):
    """Subsample a specific trajectory to have a specific number of points."""
    if len(trajectory) <= num_points:
        return pad_trajectory(trajectory, num_points)
    
    indices = [int(i) for i in torch.linspace(0, len(trajectory)-1, num_points + 1)]
    return [trajectory[i] for i in indices[1:]]

def filter_every_nth_example(examples, idx, filter_every_nth=10, processor: Optional[AutoProcessor] = None):
    """Filter every nth example."""
    return idx % filter_every_nth == 0

def process_label_batched(examples, horizon=100, num_subsampled_points=5, processor: Optional[AutoProcessor] = None):
    """Process labels for trajectory data."""
    batch_size = len(examples["horizon"])
    if isinstance(horizon, (int, float)):
        chosen_horizon = horizon * torch.ones(batch_size)
    elif isinstance(horizon, dict):
        horizon_choices = torch.Tensor([int(k) for k in horizon.keys()])
        horizon_probs = torch.Tensor([float(v) for v in horizon.values()])
        chosen_horizon_probs = torch.multinomial(horizon_probs, num_samples=batch_size, replacement=True)
        chosen_horizon = horizon_choices[chosen_horizon_probs]
    
    # Process trajectories
    trajs_2d = []
    trajs_3d = []
    for i, h in enumerate(chosen_horizon):
        max_horizon = int(min(examples['horizon'][i], h.cpu().item()))
        trajs_2d.append(examples['trajectory_2d'][i][:max_horizon])
        # Only process 3d trajectory if it exists in the huggingface dataset
        if 'trajectory_3d' in examples:
            trajs_3d.append(examples['trajectory_3d'][i][:max_horizon])
    examples["shorter_trajectory_2d"] = trajs_2d
    if 'trajectory_3d' in examples:
        examples["shorter_trajectory_3d"] = trajs_3d

    # Subsample trajectories
    examples["subsampled_trajectory_2d"] = [subsample_trajectory(traj, num_subsampled_points) for traj in trajs_2d]
    examples["goal_2d"] = [tuple(map(int, traj_2d_subsampled[-1])) for traj_2d_subsampled in examples["subsampled_trajectory_2d"]]

    return examples

def update_text_command_batched(examples, processor: Optional[AutoProcessor] = None):
    """Update text commands with goal coordinates."""
    goals = torch.Tensor(examples['goal_2d'])
    camera_param_dicts = [json.loads(camera_params) for camera_params in examples['camera_params']]
    camera_resolutions = torch.Tensor([(cam_param_dict['resolution']['width'], cam_param_dict['resolution']['height']) 
                                      for cam_param_dict in camera_param_dicts])
    
    # Normalize coordinates
    goals[:, 0] /= camera_resolutions[:, 0]
    goals[:, 1] /= camera_resolutions[:, 1]

    # Convert to integers in range [0, 1024)
    goals_tensor = (goals * 1024).to(int)

    # Convert to list of tokens
    goals_list = goals_tensor.tolist()
    goals = [point_to_token(point) for point in goals_list]

    examples['text'] = [f"Navigate to x={goal[0]}, y={goal[1]}" for goal in goals]

    return examples

def convert_coordinates_to_tokens_batched(examples, processor: Optional[AutoProcessor] = None):
    """Convert coordinates to token format for PaliGemma."""
    # Convert batch of labels to tensor
    label_tensor = torch.Tensor(examples['subsampled_trajectory_2d'])
    camera_param_dicts = [json.loads(camera_params) for camera_params in examples['camera_params']]
    camera_resolutions = torch.Tensor([(cam_param_dict['resolution']['width'], cam_param_dict['resolution']['height']) 
                                      for cam_param_dict in camera_param_dicts])
    
    # Normalize coordinates
    label_tensor[:, :, 0] /= camera_resolutions[:, [0]]  # Divide by width
    label_tensor[:, :, 1] /= camera_resolutions[:, [1]]  # Divide by height
    
    # Verify normalization
    assert torch.all(label_tensor[:, :, 0] <= 1.0) and torch.all(label_tensor[:, :, 1] <= 1.0), \
        "Image normalization is wrong. Check width/height order"
    
    # Convert to integers in range [0, 1024)
    label_tensor = (label_tensor * 1024).to(int)
    
    # Convert to list and then to tokens
    label_list = label_tensor.tolist()
    label_list = [[point_to_token(point) for point in trajectory] for trajectory in label_list]
    label_text = []
    for traj in label_list:
        traj_tokens = []
        for point_tokens in traj:
            traj_tokens.extend(point_tokens)
        label_text.append(''.join(traj_tokens))
    # Update the examples
    examples['label'] = label_text
    examples['label_list'] = label_list
    
    return examples

def compute_curvature(trajectory_2d):
    """Calculate winding factor for a trajectory. 
    
    Args:
        trajectory_2d: list of lists, where each interior list is a set of int image coordinates.

    It computes curvature based on "winding factor" which is path length/straight distance between start and finish.    
    """
    if len(trajectory_2d) < 2:
        return 1.0  # Straight line by default for very short trajectories
    
    # Convert to tensor
    points = torch.Tensor(trajectory_2d)
    
    # Calculate path length (sum of segment lengths)
    vectors = points[1:] - points[:-1]
    segment_lengths = torch.linalg.norm(vectors, dim=1)
    path_length = torch.sum(segment_lengths)
    
    # Calculate straight-line distance from start to end
    straight_distance = torch.linalg.norm(points[-1] - points[0])
    
    # Avoid division by zero
    if straight_distance < 1e-10:
        return 1.0
    
    # Winding factor is the ratio of actual path length to straight-line distance
    return float(path_length / straight_distance)

def add_curvature_to_dataset_batched(examples, processor: Optional[AutoProcessor] = None):
    # Not sure why this doesn't work
    # examples["curvature"] = [compute_curvature(example["shorter_trajectory_2d"] for example in examples)]

    examples["curvature"] = [compute_curvature(example) for example in examples['shorter_trajectory_2d']]

    return examples

def curvature_filter_transform(dataset, filter_by_curvature=False, exclude_outliers_pct=None, processor=None):
    """Filter dataset based on curvature, excluding outliers.
    
    Args:
        filter_by_curvature: Whether to filter by curvature.
        exclude_outliers_pct: Percentage of examples to exclude as outliers.
        processor: Processor to use for tokenization.

    If filter_by_curvature is True, it sorts the dataset by curvature. If exclude_outliers_pct is not None, it excludes the top percentage of examples as outliers.
    """
    if not filter_by_curvature:
        return dataset

    if exclude_outliers_pct is None or exclude_outliers_pct <= 0:
        exclude_outliers_pct = 0.0
    
    # Sort by curvature (highest to lowest)
    dataset = dataset.sort("curvature", reverse=True)
    
    # Determine indices to keep
    total_examples = len(dataset)
    
    # Calculate how many examples to exclude as outliers
    exclude_count = 0
    if exclude_outliers_pct and exclude_outliers_pct > 0:
        exclude_count = int(total_examples * exclude_outliers_pct / 100)
    
    # Select all examples except the excluded outliers
    indices_to_keep = list(range(exclude_count, total_examples))
    dataset = dataset.select(indices_to_keep)
    
    print(f"Selected {len(dataset)} samples based on curvature")
    if exclude_count > 0:
        print(f"  - Excluded top {exclude_outliers_pct}% ({exclude_count} samples) as potential outliers")
    
    return dataset

# Format to the TRL message format
def convert_to_messages(example, processor: Optional[AutoProcessor] = None):
    # Flatten tokens in the label
    label_text = ''.join([token for point in example["label"] for token in point])
    
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["text"]},
                    {"type": "image", "image": example["image"].convert('RGB')}
                ]
            },
            {
                "role": "assistant", 
                "content": [{"type": "text", "text": label_text}]
            }
        ]
    }

def collate_fn(examples, processor, device, torch_dtype=torch.bfloat16):
    """Process a batch of examples for inference."""
    texts = ["<image><bos>" + example["text"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    
    tokens = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="longest",
    )
    
    return tokens.to(torch_dtype).to(device)

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
        raise ValueError(f"Insufficient location tokens found: {trajectory_string}")
    
    # Convert tokens to coordinates
    coords = [decode_token_to_coordinates(token) for token in tokens]

    if len(coords) < expected_tokens:
        # Make sure we have an even number of coordinates (complete pairs)
        if len(coords) % 2 != 0:
            coords = coords[:-1]  # Remove the last coordinate if odd number
            
        # Find the last complete pair of coordinates
        if len(coords) >= 2:
            last_pair = coords[-2:]
            
            # Pad with the last complete pair until we have at least 10 coordinates
            while len(coords) < expected_tokens:
                coords.extend(last_pair)
    
    coords = torch.tensor(coords).reshape(-1, 2).float()
    
    # Normalize coordinates back to image space
    coords[:, 0] = coords[:, 0] * img_width / 1024
    coords[:, 1] = coords[:, 1] * img_height / 1024
    
    return coords