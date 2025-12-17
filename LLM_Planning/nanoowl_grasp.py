
import rclpy
import cv2
import numpy as np
import time
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from PIL import Image as PILImage
from nanoowl.owl_predictor import OwlPredictor

from manipulation.grasping import GraspingNode
from manipulation.kinematics import forward_kinematics

# --- CONFIGURATION ---
OBJECTS = ["red cube", "blue cube", "green cube", "bird", "chair", "horse"]
THRESHOLD = 0.1
GRASP_SCORE_THRESHOLD = 0.15

# Camera Intrinsics (Approximate Realsense D435)
FX = 607.0
FY = 607.0
CX = 320.0
CY = 240.0

def main():
    # Initialize GraspingNode (which initializes ROS)
    # Note: GraspingNode initializes rclpy.init(), so we don't need to do it here if we instantiate it first.
    # However, standard practice is rclpy.init() first. GraspingNode base calls init if not initialized?
    # Let's see GraspingNodeBase... it calls rclpy.init(). So we should NOT call it again if we want to be safe,
    # or rely on rclpy.ok() check. 
    # But for a script, let's just create the node.
    
    node = GraspingNode("nanoowl_grasping")
 #   node.align_to_init()
    
    print("Loading AI Engine... (Approx 10s)")
    # Adjust path if necessary. Assuming running from workspace root or src setup.
    # native_test.py used: "data/owl_image_encoder_patch32.engine"
    # We need to find where 'data' is relative to CWD.
    # If user runs from src/LLM_Planning, data might not be there.
    # It seems 'nanoowl' is installed or accessible.
    
    # Try absolute path or relative to nanoowl package if possible.
    # For now, let's assume the user runs this from a place where 'data' is accessible OR
    # use the path from nanoowl source if we can find it.
    # native_test.py was in src/nanoowl.
    
    predictor = OwlPredictor(
        "google/owlvit-base-patch32",
        image_encoder_engine="/home/ubuntu/ros2_ws/src/nanoowl/data/owl_image_encoder_patch32.engine"
    )

    print(f"Encoding text labels: {OBJECTS}")
    text_encodings = predictor.encode_text(OBJECTS)
    
    print("✅ Model Ready! Waiting for images...")

    try:
        while rclpy.ok():
            # Process one frame
            rclpy.spin_once(node, timeout_sec=0.1)
            
            if node.image is None or node.depth_image is None:
                continue
                
            # 1. Prepare Image
            img_bgr = node.image.copy()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = PILImage.fromarray(img_rgb)
            
            # 2. Predict
            output = predictor.predict(
                image=img_pil,
                text=OBJECTS,
                text_encodings=text_encodings,
                threshold=THRESHOLD
            )
            
            # 3. Parse Detections
            target_box = None
            target_label = None
            
            print(f"Scores: {output.scores}") 
            for i, score in enumerate(output.scores):
                label = OBJECTS[output.labels[i]]
                box = output.boxes[i]
                print(f"Found: {label} ({score:.2f})")
                
                if score > GRASP_SCORE_THRESHOLD:
                    if "cube" in label:
                        print(f"👁️ SEEN TARGET: {label} ({score:.2f}) at box={box}")
                        target_box = box
                        target_label = label
                        break # Grasp first found cube
            
            if target_box is not None:
                # 4. Calculate 3D Position
                x0, y0, x1, y1 = target_box
                u = int((x0 + x1) / 2)
                v = int((y0 + y1) / 2)
                
                # Check bounds
                h, w = node.depth_image.shape
                if 0 <= u < w and 0 <= v < h:
                    d = node.depth_image[v, u] # depth in mm or meters? 
                    # Usually 'passthrough' for Realsense 16bit is mm.
                    # Verify typical depth values. If > 100, likely mm.
                    # If < 10, likely meters.
                    # Let's assume mm and convert to meters if > 10.
                    
                    depth_m = d
                    if d > 100: 
                        depth_m = d / 1000.0
                    
                    if depth_m > 0.1 and depth_m < 2.0: # Valid range
                        print(f"Target Depth at ({u}, {v}): {depth_m:.3f}m")
                        
                        # Back-project to Camera Frame
                        # Z is depth
                        # X = (u - cx) * Z / fx
                        # Y = (v - cy) * Z / fy
                        
                        X_cam = (u - CX) * depth_m / FX
                        Y_cam = (v - CY) * depth_m / FY
                        Z_cam = depth_m
                        
                        P_cam = np.array([X_cam, Y_cam, Z_cam, 1.0])
                        
                        # 5. Transform to World Frame
                        # Get current joint positions to compute current Forward Kinematics for Camera
                        q_now = node.get_joint_positions() # This might be pulse2angle
                        T_world_cam = forward_kinematics(q_now, "cam")
                        
                        P_world = T_world_cam @ P_cam
                        
                        print(f"Target World Pos: {P_world[:3]}")
                        
                        # 6. Construct Grasp Target Pose
                        T_world_block = np.eye(4)
                        T_world_block[:3, 3] = P_world[:3]
                        # Orientation will be handled by grasp_pose (top-down)
                        
                        # 7. Execute Grasp
                        print(f"Initiating Grasp for {target_label}...")
                        success = node.grasp_pose(T_world_block)
                        
                        if success:
                            print("Grasp Successful!")
                            # Place somewhere? For now, just hold or maybe place back
                            # node.place("blue_3") # Example
                            break
                        else:
                            print("Grasp Failed.")
                    else:
                        print(f"Invalid depth: {depth_m}")
                else:
                    print("Target center out of bounds")
            
            # Sleep slightly
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
