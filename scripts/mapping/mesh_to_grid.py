import open3d as o3d
import numpy as np
import cv2
import yaml
import os
import argparse

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

def create_occupancy_grid(mesh_path, output_dir, resolution=0.05, num_points=1000000, up_axis="y",
                          inflate_m=0.0):
    print(f"Loading 3D mesh from: {mesh_path}")

    up_idx = AXIS_INDEX[up_axis]
    floor_idx_a, floor_idx_b = [i for i in (0, 1, 2) if i != up_idx]

    # 1. Load the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        print("Error: No triangles found in the mesh file.")
        return

    # 2. Convert mesh to Point Cloud by sampling points on the surface
    print(f"Sampling {num_points} points from the mesh surface...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)

    # Save as .pcd file for potential ROS usage (e.g., FAST-LIO)
    pcd_path = os.path.join(output_dir, "room_cloud.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved Point Cloud to: {pcd_path}")

    # 3. Calculate height to separate the floor from walls/obstacles.
    # up_axis selects which column is "up" (Luma AI exports are usually Y-up;
    # meshes authored directly in a robotics/world frame are often Z-up).
    # Find the lowest value on the up-axis (5th percentile to filter out noise)
    floor_level = np.percentile(points[:, up_idx], 5)

    # Define obstacle height range (taking a narrow slice higher up to avoid floor bumps/beds/chairs)
    # 80 cm to 1.3 meters above the lowest point usually catches just the walls
    obstacle_min = floor_level + 0.8
    obstacle_max = floor_level + 1.3
    obstacle_mask = (points[:, up_idx] > obstacle_min) & (points[:, up_idx] < obstacle_max)
    obstacles = points[obstacle_mask]

    if len(obstacles) == 0:
        print("Warning: No obstacles found within the specified height range.")
        return

    # 4. Project the two non-up axes to a 2D top-down grid.
    min_a, min_b = np.min(points[:, floor_idx_a]), np.min(points[:, floor_idx_b])
    max_a, max_b = np.max(points[:, floor_idx_a]), np.max(points[:, floor_idx_b])

    # Calculate grid width and height in pixels
    width = int(np.ceil((max_a - min_a) / resolution))
    height = int(np.ceil((max_b - min_b) / resolution))

    print(f"Creating map of size: {width} x {height} pixels (resolution: {resolution} m/px)")

    # Initialize a grayscale map (205 = unexplored, 0 = occupied, 254 = free space)
    # Start by setting everything as free space (254)
    grid_img = np.full((height, width), 254, dtype=np.uint8)

    # Map obstacle coordinates to pixel indices (using the two floor-plan axes)
    obs_a_idx = ((obstacles[:, floor_idx_a] - min_a) / resolution).astype(int)
    obs_b_idx = ((obstacles[:, floor_idx_b] - min_b) / resolution).astype(int)

    # Prevent index out of bounds
    obs_a_idx = np.clip(obs_a_idx, 0, width - 1)
    obs_b_idx = np.clip(obs_b_idx, 0, height - 1)

    # Draw obstacles (black = 0). We invert the second axis index to match image coordinates (top-left origin).
    grid_img[height - 1 - obs_b_idx, obs_a_idx] = 0

    # Optional inflation. Surface sampling only hits the faces of a wall, so a
    # thick wall comes out as two thin dotted outlines with a hollow middle --
    # A* can slip through a 1-cell gap and walk *inside* the wall. Growing the
    # obstacles by inflate_m closes those gaps and doubles as robot clearance.
    if inflate_m > 0:
        r = max(1, int(round(inflate_m / resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        occupied = (grid_img == 0).astype(np.uint8)
        occupied = cv2.dilate(occupied, kernel)
        grid_img[occupied == 1] = 0
        print(f"Inflated obstacles by {inflate_m} m ({r} cells)")

    # 5. Save the map image and the ROS YAML file
    img_path = os.path.join(output_dir, "room_map.png")
    cv2.imwrite(img_path, grid_img)
    print(f"Saved map image to: {img_path}")

    # Create the YAML file (standard for ROS map_server)
    yaml_path = os.path.join(output_dir, "room_map.yaml")
    yaml_data = {
        "image": "room_map.png",
        "resolution": resolution,
        "origin": [float(min_a), float(min_b), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"Saved ROS YAML file to: {yaml_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a 3D mesh (.obj) to a 2D ROS Occupancy Grid.')
    parser.add_argument('--mesh', type=str, required=True, help='Path to the input .obj file')
    parser.add_argument('--output', type=str, default='.', help='Directory to save the outputs')
    parser.add_argument('--res', type=float, default=0.05, help='Resolution in meters per pixel (default: 0.05)')
    parser.add_argument('--up-axis', type=str, default='y', choices=['x', 'y', 'z'],
                         help="Which axis is 'up' in the mesh (default: y, Luma AI convention). "
                              "Use 'z' for meshes authored in a Z-up/robotics frame.")
    parser.add_argument('--points', type=int, default=1000000,
                        help='Points sampled from the mesh surface (default: 1000000). '
                             'Raise for large meshes so walls come out solid.')
    parser.add_argument('--inflate', type=float, default=0.0,
                        help='Grow obstacles by this many meters (default: 0 = off). Closes gaps in '
                             'sampled walls and keeps routes away from them; ~robot half-width.')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    create_occupancy_grid(args.mesh, args.output, args.res, num_points=args.points,
                          up_axis=args.up_axis, inflate_m=args.inflate)
