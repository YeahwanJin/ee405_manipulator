import sys
import os
import time
import re
import ast
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from google import genai
from google.genai import types

# API Key Check
if "API_KEY" not in os.environ:
    print("!! API_KEY not found in environment variables. Please set it.")

# -------------------------------------------------------------
# [경로 설정] (아까와 동일)
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
# [Nav Class] 기존 코드 + "기다리기(Blocking)" 기능 추가
# -------------------------------------------------------------
class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

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

    def move_to_coordinate(self, x, y, z=-0.7124031163741047, w=0.7017704751415977):
        print(f"Waiting for Nav2 server... (Target: {x}, {y})")
        
        # 서버 연결 확인
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            print("Nav2 Server not available!")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = float(z)
        goal_msg.pose.pose.orientation.w = float(w)

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

# -------------------------------------------------------------
# [Main Mission] 전체 시나리오 제어
# -------------------------------------------------------------
class SmartMission:
    def __init__(self, nav_node, grasp_node):
        self.nav = nav_node  # 네비게이션 컨트롤러를 받아서 씀
        self.grasp = grasp_node
        self.world_map = {
            "loc_1": {"coords": (0.22586322614275445, -0.7200596135971158), "zone_id": None, "cube_color": None},
            "loc_2": {"coords": (-0.9819009900093079, -1.2106988430023193), "zone_id": None, "cube_color": None},
            "loc_3": {"coords": (-2.126121997833252, -1.3567204475402832), "zone_id": None, "cube_color": None}
        }
        self.task_queue = task_queue
        
        # [Import MarkerDetector]
        # Adding path to manipulation_experiment_team9 for MarkerDetector
        manip_path = os.path.abspath(os.path.join(current_dir, '../manipulation_experiment_team9'))
        if manip_path not in sys.path:
            sys.path.append(manip_path)
        
        try:
            from manipulation.marker_detector import MarkerDetector
            # Absolute path to calibration file
            calib_path = os.path.abspath(os.path.join(manip_path, 'resources/camera_calibration.npz'))
            self.detector = MarkerDetector(calib_file_path=calib_path)
            print(">> MarkerDetector initialized successfully.")
        except Exception as e:
            print(f"!! Failed to import or initialize MarkerDetector: {e}")
            self.detector = None
            
        # Init Gemini
        try:
            self.client = genai.Client(api_key=os.environ["API_KEY"])
        except:
            self.client = None

    def run_smart_mission(self):
        # [Step 1] 사용자 입력 및 LLM 계획 수립
        user_prompt = input("Enter Task Prompt: ")
        # 여기서 실제 LLM 모듈을 호출해야 함 (지금은 더미 함수 사용)
    #    self.task_queue = self.generate_plan_from_llm(user_prompt)
        
        # [Step 2] Phase 1: 똑똑한 탐색 (loc_3 생략 로직 포함)
        print("\n--- [Phase 1] Start Smart Exploration ---")
        search_order = ["loc_1", "loc_2", "loc_3"]
        
        for key in search_order:
            self.visit_and_scan(key) # 이동 및 스캔 함수 분리 추천

        possible_colors = ["red", "blue", "green"]
        active_colors = [c for c in possible_colors if c in user_prompt.lower()]
        
        # [Step 3] 추론 (Inference)
      #  self.perform_deduction(active_colors) # loc_3 정보 채우기
        
        # [Step 4] Phase 2: 실행 (Map + Task 결합)
        self.run_phase_2_execution()
    
    def visit_and_scan(self, loc_key):
        """
        특정 위치(loc_key)로 이동한 뒤, 이미지를 캡처하고 마커를 분석하여
        world_map 정보를 업데이트하는 함수.
        Zone ID 미발견 시 카메라 각도를 조정(Tilt Up)하며 재시도.
        """
        loc_info = self.world_map[loc_key]
        x, y = loc_info["coords"]
        
        # 1. 이동 (Navigation)
        print(f"\n>> [Phase 1] Moving to {loc_key} at ({x}, {y})...")
        success = self.nav.move_to_coordinate(x, y)
        if not success:
            print(f"!! Failed to move to {loc_key}. Skipping scan.")
            return
            
        # Arm Init Position (Ensure consistent start)
        if self.grasp:
            print("   Aligning arm to init pose...")
            self.grasp.align_to_init(1.5)

        # 2. 스캔 Loop (Retry Logic)
        max_retries = 2
        for attempt in range(max_retries + 1):
            print(f">> Scanning {loc_key} (Attempt {attempt+1}/{max_retries+1})...")
            time.sleep(5.0) # 로봇/카메라 안정화
            
            
            # 3. 이미지 획득
            img = self.nav.get_image()
            if img is None:
                print("!! No image received from camera.")
                continue

            found_markers_this_frame = False
            
            # 4. 마커 감지
            if self.detector:
                markers = self.detector.detect_markers_with_pose(img)
                
                if markers:
                    print(f"   Found markers IDs: {list(markers.keys())}")
                    found_markers_this_frame = True
                    
                    for mid in markers.keys():
                        # --- Cube Color Detection (5, 6, 7) ---
                        if mid == 5:
                            loc_info["cube_color"] = "red"
                            print("   -> Found Red Block (ID 5)")
                        elif mid == 6:
                            loc_info["cube_color"] = "blue"
                            print("   -> Found Blue Block (ID 6)")
                        elif mid == 7:
                            loc_info["cube_color"] = "green"
                            print("   -> Found Green Block (ID 7)")
                        
                        # --- Zone ID Detection (1, 2, 3) ---
                        if mid in [1, 2, 3]:
                            loc_info["zone_id"] = mid
                            print(f"   -> Found Zone ID {mid}")
                else:
                    print("   No markers found.")
            else:
                print("!! MarkerDetector is not initialized.")
            
            # 체크: Zone ID를 찾았는가?
            if loc_info["zone_id"] is not None:
                print("   Zone ID Confirmation: Success.")
                break
            
            # 못 찾았다면 Retry
            if attempt < max_retries:
                print("   Zone ID MISSING. Adjusting Camera Angle (Tilt Up)...")
                if self.grasp:
                    # Joint 4 (Index 3) Tilt Up (-0.2 rad approx)
                    try:
                        current_q = self.grasp.get_joint_positions()
                        target_q = current_q.copy()
                        target_q[3] -= 0.2 
                        print(f"   Move Joint 4: {current_q[3]:.2f} -> {target_q[3]:.2f}")
                        self.grasp.set_joint_positions(target_q, 1.0)
                        time.sleep(5.0)
                    except Exception as e:
                        print(f"   !! Tilt failed: {e}")
                else:
                    print("   !! GraspingNode unavailable. Cannot tilt.")
                    break
        
        # 5. 빈 공간 처리 (모든 시도 종료 후 블록 미발견 시)
        # 단, 기존에 찾았을 수도 있으니 loc_info["cube_color"] 확인
        if loc_info["cube_color"] is None:
             print("   -> No block detected after scans. Marking as 'empty_spot'.")
             loc_info["cube_color"] = "empty_spot"

        # 맵에 업데이트 반영
        self.world_map[loc_key] = loc_info
        print(f"   Updated {loc_key}: {loc_info}")

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
        print("\n--- [Phase 2] Execution Start ---")
        
        # Color -> ID Mapping
        color_map = {"red": 5, "blue": 6, "green": 7}
        # Color -> Place Action Mapping (assumed based on final.py/grasping.py)
        # Assuming we place 'blue' cube using 'blue_3' action, etc.
        place_map = {"red": "red_3", "blue": "blue_3", "green": "green_3"}
        
        # Track what we are holding
        holding_color = None
        
        for task in self.task_queue:
            action = task["action"]
            
            if action == "pick":
                color = task["target_color"]
                print(f"\n>> Task: PICK {color}")
                
                # 1. Find location in map
                target_loc = None
                for loc, info in self.world_map.items():
                    if info["cube_color"] == color:
                        target_loc = loc
                        break
                
                if target_loc:
                    coords = self.world_map[target_loc]["coords"]
                    print(f"   -> Found {color} at {target_loc} {coords}. Moving...")
                    self.nav.move_to_coordinate(*coords)
                    
                    # Grasp
                    mid = color_map.get(color)
                    if mid:
                        print(f"   -> Executing Grasp for ID {mid}...")
                        success = self.grasp.grasp(mid)
                        if success:
                            print("      Grasp Success!")
                            holding_color = color
                        else:
                            print("      Grasp Failed!")
                    else:
                        print(f"   !! Unknown Color ID for {color}")
                else:
                    print(f"   !! Error: Cannot find {color} in World Map!")

            elif action == "place":
                zone = task["target_zone"]
                print(f"\n>> Task: PLACE at Zone {zone}")
                
                if not holding_color:
                    print("   !! Error: Not holding anything to place!")
                    continue
                
                # 1. Find location of Zone
                target_loc = None
                for loc, info in self.world_map.items():
                    if info["zone_id"] == zone:
                        target_loc = loc
                        break
                        
                if target_loc:
                    coords = self.world_map[target_loc]["coords"]
                    print(f"   -> Found Zone {zone} at {target_loc} {coords}. Moving...")
                    self.nav.move_to_coordinate(*coords)
                    
                    # Place
                    action_name = place_map.get(holding_color)
                    if action_name:
                        print(f"   -> Executing Place Action '{action_name}'...")
                        success = self.grasp.place(action_name)
                        if success:
                            print("      Place Success!")
                            holding_color = None
                        else:
                            print("      Place Failed!")
                    else:
                        print(f"   !! No defined place action for holding color {holding_color}")
                else:
                    print(f"   !! Error: Cannot find Zone {zone} in World Map!")
            
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
