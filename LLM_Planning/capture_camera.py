import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('START Capturing Images')

        self.bridge = CvBridge()

        self.input_dir = '/home/ubuntu/ros2_ws/src/LLM_Planning/capture/'
       

        # Camera image Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',  # Camera topic name
            self.image_callback,
            1)
	
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Save the image
        input_filename = os.path.join(self.input_dir, f"capture.jpg")
        cv2.imwrite(str(input_filename), cv_image)
        
        # cv2.imshow('Captured Image Realtime', cv_image)
        self.get_logger().info(f"Image storage complete.{input_filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Quit requested (q). Shutting down...')
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
