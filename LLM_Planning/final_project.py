import sys
import os
import time
import re
import ast
import json
import requests
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image as PILImage
import google.generativeai as genai_sdk

# NanoOWL & ICP imports
from nanoowl.owl_predictor import OwlPredictor
from manipulation.kinematics import forward_kinematics
from manipulation.utils.icp_utils import ICPPoseEstimator

# API Key Check
if "API_KEY" not in os.environ:
    print("!! API_KEY not found in environment variables. Please set it.")

# Voice Server Configuration
VOICE_SERVER_URL = "https://localhost:5000"

def get_voice_command(timeout=60):
    """
    Wait for voice command from the voice server.
    Returns the command text or None if timeout/error.
    """
    print(f"\n🎤 Waiting for voice command... (timeout: {timeout}s)")
    print(f"   Open {VOICE_SERVER_URL} in your browser to speak commands.")
    print(f"   Or press Ctrl+C to use keyboard input.\n")
    
    start_time = time.time()
    poll_interval = 1.0  # seconds
    
    try:
        while (time.time() - start_time) < timeout:
            try:
                # verify=False for self-signed certificate
                response = requests.get(f"{VOICE_SERVER_URL}/get_command", timeout=2, verify=False)
                data = response.json()
                
                if data.get("status") == "ok" and data.get("command"):
                    text = data["command"]["text"]
                    print(f"📥 Voice Command Received: '{text}'")
                    return text
                    
            except requests.exceptions.RequestException:
                # Server not running, skip silently
                pass
            
            time.sleep(poll_interval)
        
        print("⏰ Voice command timeout.")
        return None
        
    except KeyboardInterrupt:
        print("\n⌨️  Switching to keyboard input...")
        return None

# -------------------------------------------------------------
# [경로 설정]
# -------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if src_path not in sys.path:
    sys.path.append(src_path)

#global variable
world_map = {
"loc_1": {"zone_id": None, "cube_color": None},  # 첫번째 방문할 곳
"loc_2": {"zone_id": None, "cube_color": None},  # 두번째 방문할 곳
"loc_3": {"zone_id": None, "cube_color": None}   # 안 가도 알 수 있는 곳
}
task_queue = [
{"action": "pick", "target_color": "blue"},
{"action": "place", "target_zone": 3},
{"action": "pick", "target_color": "green"},
{"action": "place", "target_zone": 2}
]
#

# -------------------------------------------------------------
# [LLM Planner] - Improved Task Planner with Dynamic Map
# -------------------------------------------------------------
LLM_SYSTEM_PROMPT = """
You are the Task Planner for a mobile manipulator robot.
Your goal is to convert a Natural Language Instruction into a JSON sequence of atomic actions.

### INPUT DATA:
You will be provided with:
1. **Map Configuration:** Which visual marker (e.g., bird, chair, horse) is in which Zone.
2. **World State:** Which block (red, green, blue) is in which Zone.
3. **Instruction:** The task to perform.

### AVAILABLE ACTIONS:
1. `navigate(target)`: Go to 'zone_1', 'zone_2', 'zone_3', or 'start_point'.
2. `pick(target)`: Pick up 'red_block', 'green_block', or 'blue_block'.
3. `place(target)`: Place the held object at a zone ('zone_1', 'zone_2', 'zone_3').
4. `say(text)`: Speak text aloud.

### LOGIC RULES:
- **Resolve Targets:** If instruction says "zone with the Bird", look at Map Configuration to find the Zone ID.
- **Conditional Logic:** If instruction says "If there is a cube...", check the World State.
- **Switching:** A "switch" or "swap" requires a temporary place location.

### OUTPUT FORMAT:
Output ONLY a valid JSON list.
Example: [{"action": "navigate", "target": "zone_2"}, {"action": "pick", "target": "red_block"}]
"""

class LLMPlanner:
    """Improved LLM Planner with dynamic map configuration support."""
    
    def __init__(self, api_key):
        genai_sdk.configure(api_key=api_key)
        self.model = genai_sdk.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=LLM_SYSTEM_PROMPT
        )
        print(">> LLMPlanner initialized with gemini-2.5-flash")
    
    def generate_plan(self, instruction, map_config, block_state):
        """
        Generate plan based on dynamic map configuration.
        
        Args:
            instruction: Natural language task instruction
            map_config: dict (e.g., {"zone_1": "horse", "zone_2": "chair", "zone_3": "bird"})
            block_state: dict (e.g., {"red_block": "zone_1", "green_block": "zone_3"})
        
        Returns:
            List of action dictionaries
        """
        prompt = f"""
### 1. DYNAMIC MAP CONFIGURATION (Visual Markers):
{json.dumps(map_config, indent=2)}

### 2. CURRENT WORLD STATE (Blocks):
{json.dumps(block_state, indent=2)}

### 3. INSTRUCTION:
"{instruction}"

Generate the execution plan JSON.
"""
        
        print(f">> LLM Planning for: '{instruction}'")
        print(f"   Map Config: {map_config}")
        print(f"   Block State: {block_state}")
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            print(f"   LLM Raw Response: {text}")
            
            # Clean up markdown code blocks if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            # Parse JSON
            plan = json.loads(text)
            print(f"   Parsed Plan: {plan}")
            return plan
            
        except json.JSONDecodeError as e:
            print(f"!! JSON Parse Error: {e}")
            # Try regex fallback
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return []
        except Exception as e:
            print(f"!! LLM Error: {e}")
            return []

# -------------------------------------------------------------
# [Nav Class] 기존 코드 + "기다리기(Blocking)" 기능 추가
# -------------------------------------------------------------
class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # [cmd_vel] Publisher for direct velocity control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # [Vision] Image Subscriber
        self.bridge = CvBridge()
        self.latest_image = None
        self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 10)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def get_image(self):
        return self.latest_image

    def move_to_coordinate(self, x, y, yaw=None):
        """
        Navigate to coordinate (x, y) facing toward the target.
        
        Args:
            x, y: Target coordinates
            yaw: Optional specific yaw angle in radians. 
                 If None, automatically faces toward target from origin.
        """
        import math
        
        # Calculate yaw to face the target from origin (0, 0)
        if yaw is None:
            yaw = math.atan2(y, x)
        
        # Convert yaw to quaternion (only z and w needed for 2D)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        
        print(f"Waiting for Nav2 server... (Target: {x}, {y}, yaw: {math.degrees(yaw):.1f}°)")
        
        # 서버 연결 확인
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            print("Nav2 Server not available!")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = float(qz)
        goal_msg.pose.pose.orientation.w = float(qw)

        print("Sending goal...")
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        
        # [핵심] 여기서 전송이 완료될 때까지 rclpy가 돕니다.
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            print('Goal rejected :(')
            return False

        print('Goal accepted! Moving...')
        
        # [핵심] 도착 결과가 나올 때까지 기다립니다 (Blocking)
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result
        print('Arrived at destination!')
        return True

    def move_forward(self, distance_m, speed=0.1):
        """
        cmd_vel을 사용하여 로봇을 지정된 거리만큼 전진시킵니다.
        MPPI 없이 직접 속도 명령을 보냅니다.
        
        Args:
            distance_m: 전진 거리 (미터)
            speed: 전진 속도 (m/s), 기본값 0.1
        """
        duration = distance_m / speed
        print(f"   >> Moving forward {distance_m*100:.1f}cm using cmd_vel (duration: {duration:.1f}s)...")
        
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)  # 20Hz
        
        # 정지
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        print("   >> Forward movement complete.")

# -------------------------------------------------------------
# [NanoOWL Detector Class] - ICP 기반 pose estimation 포함
# -------------------------------------------------------------
# Camera Intrinsics
FX = 607.0
FY = 607.0
CX = 320.0
CY = 240.0
INTRINSICS = (FX, FY, CX, CY)

OBJECTS = ["orange toy block", "red toy block", "green toy block", "blue toy block", "bird", "chair", "horse"]
THRESHOLD = 0.1
GRASP_SCORE_THRESHOLD = 0.1

class NanoOwlDetector:
    def __init__(self):
        print(">> Loading NanoOWL AI Engine...")
        self.predictor = OwlPredictor(
            "google/owlvit-base-patch32",
            image_encoder_engine="/home/ubuntu/ros2_ws/src/nanoowl/data/owl_image_encoder_patch32.engine"
        )
        
        print(f">> Encoding text labels: {OBJECTS}")
        self.text_encodings = self.predictor.encode_text(OBJECTS)
        
        # Initialize ICP Estimator
        self.icp_estimator = ICPPoseEstimator(cube_size=0.035, num_points=2000)
        
        # Color mapping from object labels (for blocks)
        self.COLOR_MAP = {
            "orange toy block": "orange",
            "red toy block": "red",
            "green toy block": "green",
            "blue toy block": "blue"
        }
        
        # Zone mapping from zone markers (bird -> zone 1, chair -> zone 2, horse -> zone 3)
        self.ZONE_MAP = {
            "bird": 1,
            "chair": 2,
            "horse": 3
        }
        print(">> NanoOwlDetector initialized successfully.")

    def detect(self, image):
        """
        NanoOWL을 사용하여 블록과 Zone 감지. 색상/Zone ID와 신뢰도 반환.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
        
        output = self.predictor.predict(
            image=img_pil,
            text=OBJECTS,
            text_encodings=self.text_encodings,
            threshold=THRESHOLD
        )
        
        detections = {
            'blocks': {},  # 블록 감지 결과
            'zones': {}    # Zone 감지 결과
        }
        
        for i, score in enumerate(output.scores):
            if score > GRASP_SCORE_THRESHOLD:
                label = OBJECTS[output.labels[i]]
                box = output.boxes[i]
                
                # 블록 감지
                if "block" in label:
                    color = self.COLOR_MAP.get(label, label)
                    print(f"   [NanoOWL] Detected {label} (score: {score:.2f}) -> {color}")
                    detections['blocks'][color] = {
                        'score': float(score),
                        'box': [int(v) for v in box]
                    }
                
                # Zone 마커 감지 (bird, chair, horse)
                elif label in self.ZONE_MAP:
                    zone_id = self.ZONE_MAP[label]
                    print(f"   [NanoOWL] Detected {label} (score: {score:.2f}) -> Zone {zone_id}")
                    detections['zones'][zone_id] = {
                        'score': float(score),
                        'marker': label,
                        'box': [int(v) for v in box]
                    }
        
        return detections

    def detect_and_get_pose(self, rgb_image, depth_image, grasp_node):
        """
        NanoOWL로 블록 감지 후 ICP를 사용하여 world frame에서의 pose 계산.
        Returns: (color, T_world_block) 또는 (None, None) if not found
        """
        img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
        
        output = self.predictor.predict(
            image=img_pil,
            text=OBJECTS,
            text_encodings=self.text_encodings,
            threshold=THRESHOLD
        )
        
        for i, score in enumerate(output.scores):
            if score > GRASP_SCORE_THRESHOLD:
                label = OBJECTS[output.labels[i]]
                box = output.boxes[i]
                
                if "block" in label:
                    color = self.COLOR_MAP.get(label, label)
                    print(f"   [NanoOWL] Found {label} (score: {score:.2f})")
                    
                    # ICP pose estimation
                    x0, y0, x1, y1 = [int(v) for v in box]
                    
                    # Add padding
                    pad = 10
                    h, w = depth_image.shape
                    x0 = max(0, x0 - pad)
                    y0 = max(0, y0 - pad)
                    x1 = min(w, x1 + pad)
                    y1 = min(h, y1 + pad)
                    
                    roi = (x0, y0, x1-x0, y1-y0)
                    
                    print("   Running ICP...")
                    scene_pcd = self.icp_estimator.depth_to_pointcloud(depth_image, INTRINSICS, roi=roi)
                    T_cam_block = self.icp_estimator.estimate_pose(scene_pcd, max_dist=0.02)
                    
                    t_cam = T_cam_block[:3, 3]
                    print(f"   ICP Refined Pos (Cam Frame): {t_cam}")
                    
                    # Transform to World
                    q_now = grasp_node.get_joint_positions()
                    T_world_cam = forward_kinematics(q_now, "cam")
                    T_world_block = T_world_cam @ T_cam_block
                    
                    print(f"   Target World Pose:\n{T_world_block}")
                    
                    return color, T_world_block
        
        return None, None


# -------------------------------------------------------------
# [Main Mission] 전체 시나리오 제어
# -------------------------------------------------------------
class SmartMission:
    def __init__(self, nav_node, grasp_node):
        self.nav = nav_node
        self.grasp = grasp_node
        import math
        # Default values used when detection fails
        self.world_map = {
            # Zone 1: Face -90° (right) - Default: zone_1, blue block
            "loc_1": {
                "coords": (1.824796199798584, -0.5843376517295837), 
                "yaw": -math.pi/2, 
                "zone_id": None, "cube_color": None, "marker": None,
                "default_zone_id": 1, "default_cube_color": "blue", "default_marker": "bird"
            },
            # Zone 2: Face 180° (backward) - Default: zone_2, green block
            "loc_2": {
                "coords": (0.526938259601593, -1.097978115081787), 
                "yaw": math.pi, 
                "zone_id": None, "cube_color": None, "marker": None,
                "default_zone_id": 2, "default_cube_color": "green", "default_marker": "chair"
            },
            # Zone 3: Face 180° (backward) - Default: zone_3, empty (red block not here)
            "loc_3": {
                "coords": (0.980643630027771, 1.1787939071655273), 
                "yaw": math.pi, 
                "zone_id": None, "cube_color": None, "marker": None,
                "default_zone_id": 3, "default_cube_color": "empty_spot", "default_marker": "horse"
            }
        }
        self.task_queue = task_queue
        
        # Dynamic state for LLM planning
        self.discovered_map = {}    # zone_id -> marker_name
        self.discovered_blocks = {} # block_name -> zone_id
        
        # [Init NanoOWL Detector]
        try:
            self.detector = NanoOwlDetector()
            print(">> NanoOwlDetector initialized successfully.")
        except Exception as e:
            print(f"!! Failed to initialize NanoOwlDetector: {e}")
            self.detector = None
            
        # Init LLM Planner (new improved version)
        try:
            self.planner = LLMPlanner(api_key=os.environ["API_KEY"])
        except Exception as e:
            print(f"!! Failed to initialize LLMPlanner: {e}")
            self.planner = None

    def run_smart_mission(self):
        # [Step 1] Get user instruction via Voice or Keyboard
        print("\n" + "="*50)
        print("🎤 VOICE INPUT MODE")
        print("="*50)
        
        # Try voice input first
        user_prompt = get_voice_command(timeout=30)
        
        # Fallback to keyboard if no voice command
        if not user_prompt:
            user_prompt = input("Enter Task Prompt (keyboard): ")
        
        # [Step 2] Phase 1: Smart Exploration
        print("\n--- [Phase 1] Start Smart Exploration ---")
        search_order = ["loc_1", "loc_2", "loc_3"]
        
        for key in search_order:
            self.visit_and_scan(key)
        
        # Build dynamic state from exploration
        self._build_dynamic_state()

        # [Step 3] Generate Plan with LLM using dynamic state
        print("\n--- [Phase 2] Generate Plan with LLM ---")
        if self.planner:
            self.task_queue = self.planner.generate_plan(
                user_prompt, 
                self.discovered_map, 
                self.discovered_blocks
            )
        else:
            print("!! LLMPlanner not available, using default queue")
        
        if not self.task_queue:
            print("!! No valid plan generated. Using default task queue.")
        else:
            print(f">> Generated Plan: {json.dumps(self.task_queue, indent=2)}")
        
        # [Step 4] Phase 3: Execution
        print("\n--- [Phase 3] Execution ---")
        self.run_phase_2_execution()
    
    def _build_dynamic_state(self):
        """Build discovered_map and discovered_blocks from world_map."""
        # Zone marker name mapping
        marker_names = {1: "bird", 2: "chair", 3: "horse"}
        
        for loc, info in self.world_map.items():
            zone_id = info.get("zone_id")
            cube_color = info.get("cube_color")
            marker = info.get("marker")
            
            if zone_id:
                zone_key = f"zone_{zone_id}"
                # Use marker name if available, otherwise use default
                if marker:
                    self.discovered_map[zone_key] = marker
                elif zone_id in marker_names:
                    self.discovered_map[zone_key] = marker_names[zone_id]
            
            if cube_color and cube_color != "empty_spot":
                block_key = f"{cube_color}_block"
                if zone_id:
                    self.discovered_blocks[block_key] = f"zone_{zone_id}"
        
        print(f"\n>> Built Dynamic State:")
        print(f"   Map Config: {self.discovered_map}")
        print(f"   Block State: {self.discovered_blocks}")
    
    def visit_and_scan(self, loc_key):
        """
        Two-stage scanning approach:
        1. Navigate to location
        2. Scan for BLOCKS in init position (good for cube detection)
        3. Tilt camera UP to scan for ZONE PICTURES (bird/chair/horse)
        4. Return to init position before navigating to next location
        """
        loc_info = self.world_map[loc_key]
        x, y = loc_info["coords"]
        yaw = loc_info.get("yaw")  # Get fixed orientation for this zone
        
        # 1. 이동 (Navigation)
        print(f"\n>> [Phase 1] Moving to {loc_key} at ({x}, {y})...")
        success = self.nav.move_to_coordinate(x, y, yaw=yaw)
        if not success:
            print(f"!! Failed to move to {loc_key}. Skipping scan.")
            return
        
        # MPPI로 도착 후 cmd_vel로 10cm 추가 전진
        print(f"   >> Moving forward 10cm closer to block...")
        try:
            self.nav.move_forward(0.3)  # 10cm = 0.1m
            time.sleep(1.0)
        except Exception as e:
            print(f"   !! Failed to move forward 10cm: {e}")
            
        # ============ STAGE 1: Scan for BLOCKS in init position ============
        if self.grasp:
            print("   [Stage 1] Aligning arm to init pose for BLOCK detection...")
            rclpy.spin_once(self.grasp, timeout_sec=0.1)
            self.grasp.align_to_init(1.5)
            time.sleep(2.0)
        
        print(f">> [Stage 1] Scanning for BLOCKS at {loc_key}...")
        time.sleep(3.0)  # Camera stabilization
        
        # Use GraspingNode's image (same as working test code)
        # Spin to get latest image
        if self.grasp:
            for _ in range(10):  # Spin multiple times to ensure fresh image
                rclpy.spin_once(self.grasp, timeout_sec=0.1)
            img = self.grasp.image
        else:
            img = self.nav.get_image()
            
        if img is not None and self.detector:
            print(f"   [Stage 1] Got image: {img.shape}")
            detections = self.detector.detect(img)
            blocks = detections.get('blocks', {})
            
            if blocks:
                print(f"   [Stage 1] Found Blocks: {list(blocks.keys())}")
                for color in blocks.keys():
                    if color in ['orange', 'red', 'green', 'blue']:
                        loc_info["cube_color"] = color
                        print(f"   -> Found Block: {color}")
            else:
                print("   [Stage 1] No blocks detected.")
        
        # ============ STAGE 2: Tilt UP to scan for ZONE PICTURES ============
        print(f"\n>> [Stage 2] Tilting camera UP for ZONE PICTURE detection...")
        
        # Store init position for later return
        q_init_saved = None
        if self.grasp:
            try:
                # ROS spin to update joint states
                rclpy.spin_once(self.grasp, timeout_sec=0.1)
                q_init_saved = self.grasp.get_joint_positions().copy()
                print(f"   Current joint positions: {q_init_saved}")
                
                # Tilt camera up: adjust multiple joints for better view
                target_q = q_init_saved.copy()
                target_q[1] += 0.3   # Joint 2: lift arm up
                target_q[2] -= 0.3   # Joint 3: tilt camera up
                target_q[3] -= 0.3   # Joint 4: additional tilt up
                
                print(f"   Moving to tilted-up pose: {target_q}")
                self.grasp.set_joint_positions(target_q, 1.5)
                
                # Wait for arm to move with spinning
                for _ in range(30):  # 3초 동안 spin
                    rclpy.spin_once(self.grasp, timeout_sec=0.1)
                
                print(f"   Arm tilt complete.")
            except Exception as e:
                print(f"   !! Failed to tilt camera up: {e}")
                import traceback
                traceback.print_exc()
        
        # Scan for zone pictures with tilted camera
        max_retries = 2
        for attempt in range(max_retries + 1):
            print(f">> [Stage 2] Scanning for ZONE PICTURE (Attempt {attempt+1}/{max_retries+1})...")
            time.sleep(3.0)
            
            # Use GraspingNode's image (same as working test code)
            if self.grasp:
                for _ in range(10):  # Spin multiple times to ensure fresh image
                    rclpy.spin_once(self.grasp, timeout_sec=0.1)
                img = self.grasp.image
            else:
                img = self.nav.get_image()
                
            if img is None:
                print("!! No image received from camera.")
                continue

            print(f"   [Stage 2] Got image: {img.shape}")
            if self.detector:
                detections = self.detector.detect(img)
                zones = detections.get('zones', {})
                
                if zones:
                    print(f"   [Stage 2] Found Zones: {list(zones.keys())}")
                    for zone_id in zones.keys():
                        if zone_id in [1, 2, 3]:
                            loc_info["zone_id"] = zone_id
                            loc_info["marker"] = detections['zones'][zone_id].get('marker')
                            print(f"   -> Found Zone: {zone_id} (marker: {loc_info['marker']})")
                else:
                    print("   [Stage 2] No zone pictures detected.")
            
            # Check: Did we find Zone ID?
            if loc_info["zone_id"] is not None:
                print("   Zone ID Confirmation: Success!")
                break
            
            # If not found, try tilting more
            if attempt < max_retries:
                print("   Zone ID MISSING. Tilting camera UP more...")
                if self.grasp:
                    try:
                        rclpy.spin_once(self.grasp, timeout_sec=0.1)
                        current_q = self.grasp.get_joint_positions()
                        target_q = current_q.copy()
                        target_q[3] -= 0.2  # Tilt up more
                        print(f"   Tilting more: {target_q}")
                        self.grasp.set_joint_positions(target_q, 1.0)
                        # Wait with spinning
                        for _ in range(20):  # 2초 동안 spin
                            rclpy.spin_once(self.grasp, timeout_sec=0.1)
                    except Exception as e:
                        print(f"   !! Tilt failed: {e}")
        
        # ============ STAGE 3: Return to INIT position before navigation ============
        print(f"\n>> [Stage 3] Returning to INIT position...")
        if self.grasp:
            try:
                rclpy.spin_once(self.grasp, timeout_sec=0.1)
                self.grasp.align_to_init(1.5)
                # Wait with spinning
                for _ in range(15):  # 1.5초 동안 spin
                    rclpy.spin_once(self.grasp, timeout_sec=0.1)
                print("   Arm returned to init position.")
            except Exception as e:
                print(f"   !! Failed to return to init: {e}")
        
        # ============ FALLBACK: Apply default values if detection failed ============
        # Zone ID fallback
        if loc_info["zone_id"] is None:
            default_zone = loc_info.get("default_zone_id")
            if default_zone:
                print(f"   ⚠️ Zone not detected. Using default: Zone {default_zone}")
                loc_info["zone_id"] = default_zone
                loc_info["marker"] = loc_info.get("default_marker")
        
        # Block color fallback
        if loc_info["cube_color"] is None:
            default_color = loc_info.get("default_cube_color")
            if default_color:
                print(f"   ⚠️ Block not detected. Using default: {default_color}")
                loc_info["cube_color"] = default_color
            else:
                print("   -> No block detected. Marking as 'empty_spot'.")
                loc_info["cube_color"] = "empty_spot"

        # Update world map
        self.world_map[loc_key] = loc_info
        print(f"   Updated {loc_key}: zone={loc_info['zone_id']}, color={loc_info['cube_color']}, marker={loc_info['marker']}")

    def generate_plan_from_llm(self, prompt):
        print(f">> LLM Generating Plan for: '{prompt}'")
        if not self.client:
            print("!! Gemini Client not initialized.")
            return []

        # Construct Prompt
        system_instruction = (
            "You are a robot task planner. "
            "The user will give a task like 'Put the blue cube in zone 3, and green in zone 2'. "
            "You must return a Python list of dictionaries representing the sequence of actions. "
            "Available actions: 'pick' (requires target_color), 'place' (requires target_zone). "
            "Colors: 'red', 'blue', 'green'. Zones: 1, 2, 3. "
            "Output Format: [{'action': 'pick', 'target_color': 'blue'}, {'action': 'place', 'target_zone': 3}, ...]. "
            "Return ONLY the list."
        )
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[system_instruction, prompt]
            )
            text_resp = response.text
            print(f"   LLM Raw Response: {text_resp}")
            
            match = re.search(r'\[.*\]', text_resp, re.DOTALL)
            if match:
                return ast.literal_eval(match.group())
            else:
                return []
        except Exception as e:
            print(f"!! LLM Error: {e}")
            return []

    def run_phase_2_execution(self):
        """Execute the task queue with support for new action format."""
        print("\n--- [Phase 2] Execution Start ---")
        
        # Place action mapping for grasp.place()
        place_action_map = {"red": "red_3", "blue": "blue_3", "green": "green_3"}
        
        # Track what we are holding
        holding_color = None
        
        for task in self.task_queue:
            action = task.get("action", "")
            target = task.get("target", "")
            
            # Handle legacy format (target_color, target_zone)
            if "target_color" in task:
                target = f"{task['target_color']}_block"
            if "target_zone" in task:
                target = f"zone_{task['target_zone']}"
            
            print(f"\n>> Executing: {action}({target})")
            
            # ============ NAVIGATE ============
            if action == "navigate":
                zone_num = self._extract_zone_number(target)
                if zone_num:
                    # Find location with this zone
                    target_loc = self._find_location_by_zone(zone_num)
                    if target_loc:
                        loc_info = self.world_map[target_loc]
                        coords = loc_info["coords"]
                        yaw = loc_info.get("yaw")
                        print(f"   -> Navigating to {target} at {coords}")
                        self.nav.move_to_coordinate(coords[0], coords[1], yaw=yaw)
                        try:
                            self.nav.move_forward(0.1)
                            time.sleep(1.0)
                        except Exception as e:
                            print(f"   !! Move forward failed: {e}")
                    else:
                        print(f"   !! Cannot find {target} in World Map!")
                elif target == "start_point":
                    print("   -> Returning to start point")
                    self.nav.move_to_coordinate(0.0, 0.0)
                else:
                    print(f"   !! Unknown navigation target: {target}")
            
            # ============ PICK ============
            elif action == "pick":
                # Extract color from target (e.g., "red_block" -> "red")
                color = target.replace("_block", "")
                print(f"   -> Picking {color} block")
                
                # Find location of this color
                target_loc = self._find_location_by_color(color)
                
                if target_loc:
                    coords = self.world_map[target_loc]["coords"]
                    print(f"   -> Found {color} at {target_loc} {coords}")
                    
                    # ICP-based Grasp
                    if self.detector and self.grasp:
                        print(f"   -> Detecting {color} block with NanoOWL + ICP...")
                        
                        rclpy.spin_once(self.grasp, timeout_sec=0.5)
                        rgb_img = self.grasp.image
                        depth_img = self.grasp.depth_image
                        
                        if rgb_img is not None and depth_img is not None:
                            detected_color, T_world_block = self.detector.detect_and_get_pose(
                                rgb_img, depth_img, self.grasp
                            )
                            
                            if T_world_block is not None:
                                print(f"   -> Executing ICP-based Grasp...")
                                success = self.grasp.grasp_pose(T_world_block)
                                if success:
                                    print("      Grasp Success!")
                                    holding_color = color
                                else:
                                    print("      Grasp Failed!")
                            else:
                                print(f"   !! Could not detect {color} block with ICP")
                        else:
                            print("   !! No camera image available for ICP")
                    else:
                        print("   !! Detector or Grasp node not available")
                else:
                    print(f"   !! Cannot find {color} in World Map!")
            
            # ============ PLACE ============
            elif action == "place":
                zone_num = self._extract_zone_number(target)
                print(f"   -> Placing at zone {zone_num}")
                
                if not holding_color:
                    print("   !! Error: Not holding anything to place!")
                    continue
                
                target_loc = self._find_location_by_zone(zone_num)
                
                if target_loc:
                    coords = self.world_map[target_loc]["coords"]
                    print(f"   -> Found zone {zone_num} at {target_loc} {coords}")
                    
                    action_name = place_action_map.get(holding_color)
                    if action_name:
                        print(f"   -> Executing Place Action '{action_name}'...")
                        success = self.grasp.place(action_name)
                        if success:
                            print("      Place Success!")
                            holding_color = None
                        else:
                            print("      Place Failed!")
                    else:
                        print(f"   !! No place action for color {holding_color}")
                else:
                    print(f"   !! Cannot find zone {zone_num} in World Map!")
            
            # ============ SAY ============
            elif action == "say":
                text = target or task.get("text", "")
                print(f"   Robot says: '{text}'")
            
            else:
                print(f"   !! Unknown action: {action}")
        
        print("\n--- Execution Complete ---")
    
    def _extract_zone_number(self, target):
        """Extract zone number from string like 'zone_1' or '1'."""
        if isinstance(target, int):
            return target
        if isinstance(target, str):
            if target.startswith("zone_"):
                try:
                    return int(target.split("_")[1])
                except:
                    pass
            try:
                return int(target)
            except:
                pass
        return None
    
    def _find_location_by_zone(self, zone_num):
        """Find location key by zone number."""
        for loc, info in self.world_map.items():
            if info.get("zone_id") == zone_num:
                return loc
        return None
    
    def _find_location_by_color(self, color):
        """Find location key by cube color."""
        for loc, info in self.world_map.items():
            if info.get("cube_color") == color:
                return loc
        return None
            
    def perform_deduction(self, active_colors):
        print(">> Performing Deduction...")
        
        # Gather State
        found_blocks = {} # loc -> color
        found_empty_locs = []
        unknown_locs = []
        
        for k, v in self.world_map.items():
            color = v.get("cube_color")
            if color and color != "empty_spot":
                found_blocks[k] = color
            elif color == "empty_spot":
                found_empty_locs.append(k)
            else:
                unknown_locs.append(k) # cube_color is None
                
        active_set = set(active_colors)
        found_colors_set = set(found_blocks.values())
        missing_colors = active_set - found_colors_set
        
        print(f"   found_blocks: {found_blocks}")
        print(f"   found_empty_locs: {found_empty_locs}")
        print(f"   unknown_locs: {unknown_locs}")
        print(f"   missing_colors: {missing_colors}")
        
        # Logic: 3 spots total. 2 Blocks. 1 Empty.
        # If we have confirmed the Empty Spot, then any Unknown spot MUST be a Block.
        
        # Case 1: We found 1 block, 1 empty spot. 1 unknown. missing 1 color.
        # -> Unknown becomes Missing Color.
        if len(found_empty_locs) >= 1 and len(unknown_locs) == 1 and len(missing_colors) == 1:
            target_loc = unknown_locs[0]
            color = list(missing_colors)[0]
            self.world_map[target_loc]["cube_color"] = color
            print(f"   [Deduction] {target_loc} must be {color} (Found Empty Spot at {found_empty_locs[0]}, so {target_loc} cannot be empty).")
        
        # Case 2: We found 2 blocks. 1 unknown.
        # -> Unknown becomes Empty. (Not strictly asked but good to have)
        elif len(found_blocks) == 2 and len(unknown_locs) == 1:
            target_loc = unknown_locs[0]
            self.world_map[target_loc]["cube_color"] = "empty_spot"
            print(f"   [Deduction] {target_loc} must be the Empty Spot (Found both blocks).")

        # 2. Zone ID Deduction (Same elimination logic)
        all_zones = {1, 2, 3}
        found_zones = set()
        locs_without_zone = []
        
        for k, v in self.world_map.items():
            if v["zone_id"]:
                found_zones.add(v["zone_id"])
            else:
                locs_without_zone.append(k)
                
        missing_zones = all_zones - found_zones
        
        if len(missing_zones) == 1 and len(locs_without_zone) == 1:
            target_loc = locs_without_zone[0]
            zid = list(missing_zones)[0]
            self.world_map[target_loc]["zone_id"] = zid
            print(f"   [Deduction] {target_loc} must be Zone {zid} (Elimination).")
        
        print("Final World Map:")
        for k, v in self.world_map.items():
            print(f"{k}: {v}")

# -------------------------------------------------------------
# [실행부]
# -------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    # 1. 네비게이션 노드 생성
    nav_controller = NavigationController()
    
    # 2. Grasping 노드 생성
    # Import GraspingNode dynamically
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        manip_path = os.path.abspath(os.path.join(current_dir, '../manipulation_experiment_team9'))
        if manip_path not in sys.path:
            sys.path.append(manip_path)
        from manipulation.grasping import GraspingNode
        grasp_controller = GraspingNode("grasping_node")
    except Exception as e:
        print(f"!! Failed to import GraspingNode: {e}")
        grasp_controller = None

    # 3. 미션 수행 객체 생성 (네비게이션 노드, 그랩 노드 전달)
    mission = SmartMission(nav_controller, grasp_controller)

    try:
        # 미션 시작
        mission.run_smart_mission()
        
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 처리
        nav_controller.destroy_node()
        if grasp_controller:
            grasp_controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
