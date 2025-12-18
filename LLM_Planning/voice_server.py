#!/usr/bin/env python3
"""
Voice Command Server for Robot Control
=======================================
Flask server that receives voice commands from browser and publishes to ROS2.

Usage:
    python3 voice_server.py

Then open http://ROBOT_IP:5000 in browser to use voice control.
"""

import os
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Global command storage (shared between Flask and ROS2)
latest_command = {"text": None, "timestamp": None}
command_lock = threading.Lock()


class VoiceCommandPublisher(Node):
    """ROS2 Node that publishes voice commands."""
    
    def __init__(self):
        super().__init__('voice_command_publisher')
        self.publisher = self.create_publisher(String, '/voice_command', 10)
        self.get_logger().info('Voice Command Publisher initialized')
    
    def publish_command(self, text):
        msg = String()
        msg.data = text
        self.publisher.publish(msg)
        self.get_logger().info(f'Published voice command: {text}')


# Flask App
app = Flask(__name__, static_folder='resources')
CORS(app)  # Allow cross-origin requests

ros_node = None


@app.route('/')
def index():
    """Serve the voice control HTML page."""
    return send_from_directory('resources', 'voice_control.html')


@app.route('/voice_command', methods=['POST'])
def receive_voice_command():
    """Receive voice command from browser and publish to ROS2."""
    global latest_command
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400
    
    print(f"\n🎤 Voice Command Received: '{text}'")
    
    # Store command
    with command_lock:
        import time
        latest_command = {"text": text, "timestamp": time.time()}
    
    # Publish to ROS2
    if ros_node:
        ros_node.publish_command(text)
    
    return jsonify({"status": "ok", "received": text})


@app.route('/get_command', methods=['GET'])
def get_latest_command():
    """Get the latest voice command (for polling from final_project.py)."""
    global latest_command
    
    with command_lock:
        if latest_command["text"]:
            cmd = latest_command.copy()
            latest_command = {"text": None, "timestamp": None}  # Clear after reading
            return jsonify({"status": "ok", "command": cmd})
        else:
            return jsonify({"status": "waiting", "command": None})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "ros2": ros_node is not None})


def run_flask():
    """Run Flask server with HTTPS for voice input support."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(script_dir, 'cert.pem')
    key_path = os.path.join(script_dir, 'key.pem')
    
    print("\n" + "="*50)
    print("🎤 Voice Command Server Started (HTTPS)")
    print("="*50)
    print(f"📍 Open in browser: https://YOUR_ROBOT_IP:5000")
    print(f"⚠️  Accept the security warning in browser")
    print("="*50 + "\n")
    
    # Use HTTPS if certificates exist
    if os.path.exists(cert_path) and os.path.exists(key_path):
        app.run(host='0.0.0.0', port=5000, threaded=True, 
                ssl_context=(cert_path, key_path))
    else:
        print("⚠️ SSL certificates not found, running HTTP only")
        app.run(host='0.0.0.0', port=5000, threaded=True)


def main():
    global ros_node
    
    # Initialize ROS2
    rclpy.init()
    ros_node = VoiceCommandPublisher()
    
    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    try:
        # Keep ROS2 spinning
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down voice server...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
