import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

# Setup paths (same as final_project.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.abspath(os.path.join(current_dir, '../yolo_detector'))
if yolo_path not in sys.path:
    sys.path.append(yolo_path)

try:
    from yolo_detector.yolov1_inference import Yolo, inference_image
    print(">> YOLO Module imported successfully.")
except ImportError:
    try:
         sys.path.append(os.path.join(yolo_path, 'yolo_detector'))
         from yolov1_inference import Yolo, inference_image
         print(">> YOLO Module imported (fallback).")
    except Exception as e:
        print(f"!! Failed to import YOLO: {e}")
        sys.exit(1)

class CameraCheckNode(Node):
    def __init__(self):
        super().__init__('camera_check_node')
        self.get_logger().info("Initializing Camera & YOLO Check Node...")
        
        self.bridge = CvBridge()
        
        # Load Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using Device: {self.device}")
        
        self.model = Yolo(grid_size=7, num_boxes=2, num_classes=20).to(self.device)
        
        ckpt_path = os.path.abspath(os.path.join(yolo_path, 'checkpoints/best.pth'))
        self.get_logger().info(f"Loading Checkpoint: {ckpt_path}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

        self.VOC_CLASSES = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        
        # Subscriber
        self.create_subscription(
            Image, 
            '/depth_cam/rgb/image_raw', 
            self.image_callback, 
            10
        )
        self.get_logger().info("Subscribed to /depth_cam/rgb/image_raw")
        self.get_logger().info("Press 'q' in the OpenCV window to exit.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")
            return
            
        if self.model:
            # Run Inference and Draw Boxes
            # inference_image handles resizing/normalization internally and returns image with boxes
            res_image = inference_image(
                self.model, 
                cv_image, 
                self.device, 
                self.VOC_CLASSES
            )
            
            # Add Mapping Info Overlay
            cv2.putText(res_image, "Mapping Hint:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(res_image, "Aeroplane -> Red", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(res_image, "Bicycle -> Blue", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(res_image, "Bird -> Green", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Camera + YOLO Check", res_image)
        else:
            cv2.imshow("Camera Check (No Model)", cv_image)
            
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            rclpy.shutdown()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CameraCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
