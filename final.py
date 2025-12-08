import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose  #

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        # Nav2의 navigate_to_pose 액션 서버와 연결
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def move_to_coordinate(self, x, y, w=1.0):
        """
        x, y: 이동할 목표 좌표 (meter)
        w: 도착 후 바라볼 방향 (1.0 = 초기 방향 유지/동쪽, 0.0 ~ 1.0 사이 값)
        """
        print(f"Waiting for Nav2 server... (Target: x={x}, y={y})")
        self._action_client.wait_for_server()  # 서버가 준비될 때까지 대기

        # 목표 메시지 생성
        goal_msg = NavigateToPose.Goal()
        
        # 좌표계 설정 (보통 'map' 기준)
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg() #

        # 목표 위치 입력
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0  # 2D 주행이므로 0

        # 목표 방향 입력 (Quaternion)
        # w=1.0은 맵 기준 0도(보통 오른쪽/동쪽)를 바라보는 것입니다.
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = 0.0
        goal_msg.pose.pose.orientation.w = w 

        print("Sending goal...")
        # 비동기(Async)로 목표 전송
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        
        # 전송 후 결과를 기다림 (블로킹 방식을 원하면 아래 주석 해제)
        # rclpy.spin_until_future_complete(self, self._send_goal_future)
        print("Goal Sent!")

# --- 사용 예시 (Main 문에 넣어서 테스트) ---
def main(args=None):
    rclpy.init(args=args)
    
    # 1. 네비게이션 컨트롤러 생성
    navigator = NavigationController()

    # 2. 아까 터