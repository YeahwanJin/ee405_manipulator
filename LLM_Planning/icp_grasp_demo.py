
import rclpy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from PIL import Image as PILImage
from nanoowl.owl_predictor import OwlPredictor
from rclpy.qos import qos_profile_sensor_data

from manipulation.grasping import GraspingNode
from manipulation.kinematics import forward_kinematics
from manipulation.utils.icp_utils import ICPPoseEstimator

# --- CONFIGURATION ---
OBJECTS = ["orange toy block", "red toy block", "green toy block", "blue toy block", "bird", "chair", "horse", "person"]
THRESHOLD = 0.1
GRASP_SCORE_THRESHOLD = 0.1

# Camera Intrinsics 
FX = 607.0
FY = 607.0
CX = 320.0
CY = 240.0
INTRINSICS = (FX, FY, CX, CY)

def main():
    node = GraspingNode("icp_grasping")
    # node.align_to_init() # Optional, depending on user need
    
    print("Loading AI Engine...")
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine="/home/ubuntu/ros2_ws/src/nanoowl/data/owl_image_encoder_patch32.engine"
    )

    print(f"Encoding text labels: {OBJECTS}")
    text_encodings = predictor.encode_text(OBJECTS)
    
    # Initialize ICP Estimator (Assuming 5cm cube)
    icp_estimator = ICPPoseEstimator(cube_size=0.035, num_points=2000) # Slightly smaller than Real 5cm to grasp inside? Or exact. 
    # Let's say 3.5cm - 5cm.
    
    print("✅ System Ready! ")

    try:
        while rclpy.ok():
        #    print("Waiting for image...")
            rclpy.spin_once(node, timeout_sec=0.1)
            
            if node.image is None or node.depth_image is None:
                continue
                
            img_bgr = node.image.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = PILImage.fromarray(img_rgb)
            depth_img = node.depth_image # Assuming mm
            
            # Predict
            output = predictor.predict(
                image=img_pil,
                text=OBJECTS,
                text_encodings=text_encodings,
                threshold=THRESHOLD
            )
            
            target_box = None
            target_label = None
            
            for i, score in enumerate(output.scores):
                if score > GRASP_SCORE_THRESHOLD:
                    label = OBJECTS[output.labels[i]]
                    box = output.boxes[i]
                    if "block" in label:
                        print(f"Found {label}, Score: {score}")
                        target_box = box
                        target_label = label
                        break 
            
            if target_box is not None:
                # 1. Define ROI
                x0, y0, x1, y1 = [int(v) for v in target_box]
                
                # Add padding
                pad = 10
                h, w = depth_img.shape
                x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
                x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
                
                roi = (x0, y0, x1-x0, y1-y0)
                
                # 2. Get Point Cloud from ROI
                print("Running ICP...")
                scene_pcd = icp_estimator.depth_to_pointcloud(depth_img, INTRINSICS, roi=roi)
                
                # 3. Estimate Pose
                # T_cam_block (Transformation from Model(Cube) Frame to Camera Frame)
                T_cam_block = icp_estimator.estimate_pose(scene_pcd, max_dist=0.02)
                
                # Extract Translation
                t_cam = T_cam_block[:3, 3]
                print(f"ICP Refined Pos (Cam Frame): {t_cam}")
                
                # 4. Transform to World
                q_now = node.get_joint_positions() 
                T_world_cam = forward_kinematics(q_now, "cam")
                
                T_world_block = T_world_cam @ T_cam_block
                
                print(f"Target World Pose:\n{T_world_block}")
                
                # 5. Grasp
                # We can now use the full orientation from ICP!
                # However, GraspingNode.grasp_pose might still override orientation for top-down.
                # If we want to use ICP orientation, we should pass it.
                # For now, let's just pass the position and let grasp_pose handle top-down.
                # Or modify grasp_pose to respect orientation.
                
                # Let's override the orientation to be pure top-down for safety first, 
                # but using the refined position.
                
                # Refined Position
                node.grasp_pose(T_world_block) 
                
                break 
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
