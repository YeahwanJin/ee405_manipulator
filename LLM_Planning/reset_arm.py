
import rclpy
import time
from manipulation.grasping import GraspingNode

def main():
    # Initialize the node
    # GraspingNodeBase calls rclpy.init(), so we can just instantiate
    node = GraspingNode("reset_arm_node")
    
    print("Aligning robot to initial position...")
    node.align_to_init(duration=2.0)
    
    # Wait a bit to ensure command is sent and executed
    time.sleep(2.5)
    
    print("Reset complete.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
