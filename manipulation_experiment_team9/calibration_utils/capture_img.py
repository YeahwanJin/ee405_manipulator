#!/usr/bin/env python3
"""
ROS2 Image Saver (from queue): press ENTER to save the latest frame.
- Subscribes to a sensor_msgs/Image topic
- Maintains an internal queue of recent frames
- Shows a preview window and saves when ENTER is pressed
- Press Q or ESC to quit

Usage:
  python save_on_enter_ros.py \
    --topic /camera/image_raw \
    --out ./captures \
    --prefix img \
    --ext png \
    --queue-size 5 \
    --width 0 --height 0

Notes:
- Requires: rclpy, cv_bridge, sensor_msgs, OpenCV (cv2)
- If your camera publishes a different encoding, adjust desired_encoding.
"""

import queue

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageQueueSubscriber(Node):
    def __init__(self, 
                 topic: str = "/depth_cam/rgb/image_raw", 
                 queue_size: int = 5, 
                 desired_encoding: str = "bgr8"):
        
        super().__init__("image_queue_subscriber")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, queue_size)
        )

        self.bridge = CvBridge()
        self.desired_encoding = desired_encoding
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self.latest = None  # type: ignore

        self.sub = self.create_subscription(Image, topic, self.img_callback, qos)
        self.get_logger().info(f"Subscribed to: {topic}")

    def img_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.desired_encoding)
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge error: {e}")
            return

        self.latest = img
        # put into queue; if full, drop the oldest to keep things moving
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(img)
        except queue.Full:
            pass