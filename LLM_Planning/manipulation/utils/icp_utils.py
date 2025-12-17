
import numpy as np
import open3d as o3d
import cv2

class ICPPoseEstimator:
    def __init__(self, cube_size=0.05, num_points=1000):
        """
        cube_size: edge length of the cube in meters (e.g. 0.05 for 5cm)
        """
        self.cube_size = cube_size
        self.target_cloud = self._generate_cube_pcd(cube_size, num_points)
        
    def _generate_cube_pcd(self, size, num_points):
        """Generates a synthetic point cloud for a cube centered at origin"""
        # Create a mesh and sample points
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        # Center the box (create_box puts corner at 0,0,0)
        mesh.translate(np.array([-size/2, -size/2, -size/2]))
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return pcd

    def depth_to_pointcloud(self, depth_img, intrinsics, depth_scale=1000.0, roi=None):
        fx, fy, cx, cy = intrinsics
        
        # Crop logic...
        if roi is not None:
            u_start, v_start, w, h = roi
            depth_crop = depth_img[v_start:v_start+h, u_start:u_start+w]
            cx = cx - u_start
            cy = cy - v_start
        else:
            depth_crop = depth_img

        # 1. Force Float32 (Fixes the Crash)
        depth_crop = np.ascontiguousarray(depth_crop).astype(np.float32)

        # 2. FORCE METERS (Fixes the Scale)
        # We manually divide by 1000.0 right here.
        # We don't ask Open3D to do it.
        if np.max(depth_crop) > 100.0:
            print(f"[ICP] Auto-converting MM to Meters (Max val: {np.max(depth_crop)})")
            depth_crop /= 1000.0  # <--- THE CRITICAL FIX
        
        # 3. Create Cloud
        # We set depth_scale=1.0 because we already did the division above.
        o3d_img = o3d.geometry.Image(depth_crop)
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(depth_crop.shape[1], depth_crop.shape[0], fx, fy, cx, cy)
        
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_img, 
            intrinsic_matrix, 
            depth_scale=1.0, # <--- Set to 1.0
            depth_trunc=3.0, 
            stride=1
        )
        return pcd

    def estimate_pose(self, scene_pcd, max_dist=0.02):
        """
        Run ICP to align the synthetic cube (source) to the scene (target).
        Returns: 4x4 Transformation Matrix
        """
        if len(scene_pcd.points) < 10:
            print("[ICP] Not enough points in scene cloud.")
            return np.eye(4)

        # 1. Initial Guess: Translate source centroid to scene centroid
        source = self.target_cloud
        target = scene_pcd
        
        source_center = source.get_center()
        target_center = target.get_center()
        
        # Initial transformation (Just translation)
        trans_init = np.eye(4)
        trans_init[:3, 3] = target_center - source_center
        
        # 2. Run ICP (Point-to-Point)
        # Refines the alignment
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, max_dist, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Resulting transform is from Model Space -> Camera Space
        return reg_p2p.transformation
