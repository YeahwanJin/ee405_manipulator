#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('visualization_path')

        # Subscribers
        self.create_subscription(Path, '/plan', self.plan_callback, 10)
        self.create_subscription(Path, '/local_plan', self.trajectory_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 50)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.path_pub = self.create_publisher(Path, '/executed_path', 10)

        # Store paths
        self.executed_path = Path()
        self.executed_path.header.frame_id = 'odom'  # Adjust if your TF tree uses 'odom'

        self.goal_reached = False  # Flag to clear path after reaching goal

    def plan_callback(self, msg: Path):
        # Planned path in red
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'planned_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.a = 0.7
        marker.points = [pose.pose.position for pose in msg.poses]
        self.marker_pub.publish(marker)

    def trajectory_callback(self, msg: Path):
        # Planned trajectory in blue (temporary)
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'planned_trajectory'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.b = 1.0
        marker.color.a = 0.7
        marker.lifetime = Duration(sec=1)  # lasts 1 sec only
        marker.points = [pose.pose.position for pose in msg.poses]
        self.marker_pub.publish(marker)

    def odom_callback(self, msg: Odometry):
        if self.goal_reached:
            return

        # Convert odom to PoseStamped and append to executed_path
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.executed_path.poses.append(pose_stamped)
        self.executed_path.header.stamp = self.get_clock().now().to_msg()

        # Publish accumulated executed path (gray)
        path_marker = Marker()
        path_marker.header.frame_id = self.executed_path.header.frame_id
        path_marker.ns = 'executed_path'
        path_marker.id = 2
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color.r = 0.5
        path_marker.color.g = 0.5
        path_marker.color.b = 0.5
        path_marker.color.a = 0.7
        path_marker.points = [pose.pose.position for pose in self.executed_path.poses]

        self.marker_pub.publish(path_marker)
        self.path_pub.publish(self.executed_path)

def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
