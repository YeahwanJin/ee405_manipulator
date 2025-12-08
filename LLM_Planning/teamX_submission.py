import os
import sys
import re
import time

from google import genai
from google.genai import types

import rclpy
from rclpy.node import Node

sys.path.append('/home/ubuntu/ros2_ws/src/manipulation_experiment_team9')
from run_action_group import ActionGroupExecution

API_KEY = os.environ['API_KEY']
client = genai.Client(api_key=API_KEY)

import ast # 문자열을 리스트로 안전하게 변환하기 위해 추가

def parse_gemini_response(response_text):
    # 정규표현식으로 대괄호 [...] 안에 있는 내용만 추출
    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if match:
        try:
            # 문자열 형태의 리스트("['a', 'b']")를 실제 리스트 객체(['a', 'b'])로 변환
            return ast.literal_eval(match.group())
        except:
            print("Error parsing list from response")
            return []
    return []

def get_action_from_gemini(image_path, prompt):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    client = genai.Client(api_key=API_KEY)

    image = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')   # png -> mime_type='images/png'

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[image,prompt]
    )

    return response.text

def main():
    rclpy.init()
    node = ActionGroupExecution("action_group_execution")
    image_path = '/home/ubuntu/ros2_ws/src/LLM_Planning/capture/capture.jpg'

    # [기본 설정]
    # 로봇이 수행 가능한 모든 동작 리스트 (Gemini에게 알려주기 위함)
    available_actions = (
        "Available actions: "
        "['look_left', 'look_center', 'look_right', "
        "'pick_left', 'pick_center', 'pick_right', "
        "'place_left', 'place_center', 'place_right']"
    )

    # Phase 1: scan the environment
    environment_description = "Environment Description: There is a mobile robot equipped with a robotic arm and a gripper. Three colored cubes are positioned to the left, center, and right of the robot."
    task_goal = "Task Goal: Pick up and place all the cubes in front of the robot."
    
    # ---------------------------------------------------------
    # [STEP 1] 초기 정찰 계획 수립 (output_prompt 작성)
    # ---------------------------------------------------------
    # 목표: 큐브가 어디 있는지 모르니 일단 왼쪽, 중앙, 오른쪽을 다 보게 시킨다.
    output_prompt = (
        "To identify the location and color of the cubes, I need to look around. "
        "Provide a Python list of actions to look at the left, center, and right directions sequentially. "
        "Format: ['action1', 'action2', ...]. " 
        + available_actions
    )
    
    prompt = environment_description + '\n' + task_goal + '\n' + output_prompt

    print("Asking Gemini (Phase 1 Plan)...")
    response = get_action_from_gemini(image_path, prompt)
    print(f"Gemini Response: {response}")

    # [Parsing] 텍스트에서 리스트 추출
    action_list = parse_gemini_response(response)

    analysis_response = "Environment analysis: "
    
    # ---------------------------------------------------------
    # [STEP 2] 정찰 실행 및 상황 분석 (Loop)
    # ---------------------------------------------------------
    for action_name in action_list: 
        node.execute_action_group(action_name)
        node.get_logger().info(f"Action group '{action_name}' execution finished.")
        time.sleep(1.0)

        # 목표: 방금 찍은 사진(capture.jpg)을 보고 '어느 위치'에 '무슨 색'이 있는지 텍스트로 저장한다.
        # 중요: 나중에 '파란 것과 빨간 것을 바꿔라' 같은 미션을 위해 '색상' 정보가 필수적임.
        output_prompt = (
            f"The robot just performed the action '{action_name}'. "
            "Describe the colored cube visible in this image. "
            "Specify the color (Red, Blue, Green, etc.) and confirm its position based on the action taken. "
            "Keep it concise (e.g., 'Found Red cube at Left')."
        )
        
        prompt = environment_description + '\n' + task_goal + '\n' + output_prompt

        print(f"Asking Gemini (Analyzing {action_name})...")
        response = get_action_from_gemini(image_path, prompt)
        
        # 분석 내용을 누적해서 저장 (이게 로봇의 단기 기억이 됨)
        analysis_response += f"\n- Observation after {action_name}: {response}"

    print(f"Final Analysis: {analysis_response}")

    # ---------------------------------------------------------
    # [STEP 3] 최종 행동 계획 (Phase 2)
    # ---------------------------------------------------------
    # 목표: 누적된 분석 정보(analysis_response)와 목표(task_goal)를 보고 실제 작업 순서를 짠다.
    output_prompt = (
        "Based on the 'Environment analysis' above, generate a sequence of actions to achieve the 'Task Goal'. "
        "Think logically: You must 'pick' a cube from its detected location and then 'place' it. "
        "Repeat this for all detected cubes. "
        "Return ONLY a Python list of strings. Do not include markdown or explanations. "
        f"{available_actions}"
    )
    
    prompt = environment_description + '\n' + task_goal + '\n' + analysis_response + '\n' + output_prompt

    print("Asking Gemini (Phase 2 Execution Plan)...")
    response = get_action_from_gemini(image_path, prompt)
    print(f"Gemini Response: {response}")

    # [Parsing]
    action_list = parse_gemini_response(response)

    for action_name in action_list: 
        node.execute_action_group(action_name)
        node.get_logger().info(f"Action '{action_name}' execution finished.")
        time.sleep(1.0)


if __name__=='__main__':
    main()
