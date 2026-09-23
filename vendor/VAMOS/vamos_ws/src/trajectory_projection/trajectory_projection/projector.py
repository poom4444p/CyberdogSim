import numpy as np
import cv2

class Projector:
    def __init__(self, K=None, img_height=None, img_width=None):
        self.K = K
        self.img_height = img_height
        self.img_width = img_width
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        if K is not None:
            self.set_intrinsics(K)

        if img_height is not None and img_width is not None:
            self.set_height_width(img_height, img_width)

    def set_intrinsics(self, K):
        self.K = np.array(K).reshape(3, 3)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def set_height_width(self, height, width):
        self.img_height = height
        self.img_width = width

    def _inpaint_depth(self, depth_image):
        depth = depth_image.copy().astype(np.float32)
        mask = np.isnan(depth) | (depth <= 0)
        mask = mask.astype(np.uint8)
        depth[np.isnan(depth)] = 0
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        inpainted = cv2.inpaint(depth_norm, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        inpainted = inpainted.astype(np.float32)
        inpainted = cv2.normalize(inpainted, None, depth.min(), depth.max(), cv2.NORM_MINMAX)
        return inpainted

    def get_3d_points(self, pixel_coords, depth_image):
        '''
        Args:
            pixel_coords: Nx2 numpy array of pixel coordinates
            depth_image = numpy array of shape (height, width) 
        Returns:
            Nx3 numpy array corresponding to the 3d coordinates associated with the input pixel coordinates
        '''
        pixel_coords = np.asarray(pixel_coords)
        assert pixel_coords.shape[1] == 2, "Input must be Nx2 array of pixel coordinates"
        inpainted_depth = self._inpaint_depth(depth_image)
        x_pixels = pixel_coords[:, 0]
        y_pixels = pixel_coords[:, 1]
        x_int = x_pixels.astype(int)
        y_int = y_pixels.astype(int)
        depth_values = inpainted_depth[y_int, x_int]
        valid_mask = depth_values > 0
        x_pixels = x_pixels[valid_mask]
        y_pixels = y_pixels[valid_mask]
        depth_values = depth_values[valid_mask]
        x_cam = (x_pixels - self.cx) * depth_values / self.fx
        y_cam = (y_pixels - self.cy) * depth_values / self.fy
        z_cam = depth_values
        return np.stack((x_cam, y_cam, z_cam), axis=-1)

    def project_points_to_image(self, points_3d):
        """Project 3D points to 2D image coordinates using camera intrinsics
        
        Args:
            points_3d: numpy array of shape (N, 3) containing N 3D points
            
        Returns:
            numpy array of shape (N, 2) containing pixel coordinates
            Points behind the camera will have coordinates (-1, -1)
        """
        # Check input shape
        assert points_3d.ndim == 2 and points_3d.shape[1] == 3, "Input must be an array of shape (N, 3)"
        
        # Create output array (initialize with -1 for invalid points)
        N = points_3d.shape[0]
        pixel_coords = np.full((N, 2), -1, dtype=int)
        
        # Get x, y, z components
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        
        # Filter points that are in front of the camera (z > 0)
        valid_mask = z > 0
        
        # Project valid points
        if np.any(valid_mask):
            # Perspective division
            x_normalized = x[valid_mask] / z[valid_mask]
            y_normalized = y[valid_mask] / z[valid_mask]
            
            # Apply camera intrinsics
            x_pixel = x_normalized * self.fx + self.cx
            y_pixel = y_normalized * self.fy + self.cy
            
            # Convert to integers
            pixel_coords[valid_mask, 0] = np.round(x_pixel).astype(int)
            pixel_coords[valid_mask, 1] = np.round(y_pixel).astype(int)
        
        return pixel_coords
    

    def is_point_in_image(self, points_2d, threshold=0.50):
        """
        Check whether 2D image points are visible in the image or not.
        
        Args:
            points_2d: numpy array of shape (N, 2) containing pixel coordinates (x, y)
        
        Returns:
            numpy array of shape (N,) containing boolean values
            True if point is in image, False otherwise
        """
        # Check input shape
        assert points_2d.ndim == 2 and points_2d.shape[1] == 2, "Input must be an array of shape (N, 2)"
        
        # Get x and y coordinates
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        
        # Check if points are within image bounds
        # Note: x corresponds to width (columns) and y corresponds to height (rows)
        valid_x = (x >= 0) & (x < self.img_width)
        valid_y = (y >= 0) & (y < self.img_height)
        # img_center_x = self.img_width //2 
        # threshold = threshold * self.img_width
        # valid_x = (x >= img_center_x-threshold) & (x < img_center_x+threshold)
        # valid_y = (y >= 0) & (y < self.img_height)
        
        # Point is valid if both x and y are within bounds
        valid_points = valid_x & valid_y
        
        return valid_points

    # def pixel_to_ground(self, u, v, T_world_cam, z_ground):
    #     """
    #     Project pixel (u,v) down onto ground plane z=z_ground.
    #     Returns a length-3 numpy array in world FLU coordinates.
    #     """
    #     # 1) ray in cam frame
    #     pix_h = np.array([u, v, 1.0])
    #     d_cam = np.linalg.inv(self.K) @ pix_h

    #     # 2) world‐from‐cam rotation & origin
    #     R = T_world_cam[:3, :3]
    #     O = T_world_cam[:3,  3]

    #     # 3) ray direction in world
    #     D = R @ d_cam

    #     # 4) solve for intersection with z = z_ground
    #     s = (z_ground - O[2]) / D[2]

    #     # 5) compute and return point
    #     return O + s * D

    def pixels_to_ground(self, pixel_coords, T_world_cam, z_ground):
        """
        Vectorized: project a batch of pixels onto the ground plane.

        Args:
            pixel_coords : (N,2) array of (u,v) image coordinates
            T_world_cam  : (4,4) homogenous cam→world transform
            z_ground     : scalar ground-plane height in world Z

        Returns:
            (N,3) array of 3D points in world frame
        """
        # 1) unpack and build homogeneous pixel matrix: shape (3, N)
        pix = np.asarray(pixel_coords)
        assert pix.ndim == 2 and pix.shape[1] == 2
        u = pix[:, 0]
        v = pix[:, 1]
        ones = np.ones_like(u)
        pix_h = np.stack([u, v, ones], axis=0)   # (3, N)

        # 2) back-project to camera ray: d_cam = K⁻¹ @ pix_h  → (3, N)
        Kinv = np.linalg.inv(self.K)
        d_cam = Kinv @ pix_h

        # 3) world rotation & origin
        R = T_world_cam[:3, :3]    # (3,3)
        O = T_world_cam[:3,  3]    # (3,)

        # 4) rotate all ray directions into world: D = R @ d_cam  → (3, N)
        D = R @ d_cam

        # 5) compute all s parameters:  s[i] = (z_ground - O_z) / D_z[i]
        Dz = D[2, :]
        # avoid division by zero
        valid = Dz != 0
        s = np.empty_like(Dz)
        s[valid] = (z_ground - O[2]) / Dz[valid]
        s[~valid] = np.nan

        # 6) form all intersection points P = O[:,None] + D * s[None,:]  → (3, N)
        P = O[:, None] + D * s[None, :]

        # 7) return as (N,3)
        return P.T