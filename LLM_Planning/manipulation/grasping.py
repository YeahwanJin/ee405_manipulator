import time
import cv2
import rclpy
import numpy as np

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase
from .marker_detector import MarkerDetectionResult


class GraspingNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)
        self.running = True
        
        # [j1, j2, j3, j4, j5]
        # Using same values as manipulation_experiment_team9
        self.q_init = np.array([0.0, -1.2, 1.7, 1.3, 0.0], dtype=float)

        # 필요하면 “내가 마지막에 보낸 값”만 추적하고 쓸 수도 있음
        self.last_q = None

        self.grasping = False
        self.grasp_targets = []

    # =====================================================================
    # 기본 유틸 (수정하지 말라고 하던 부분)
    # =====================================================================
    def gripper_close(self, duration=1.5):
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.5):
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        """원 코드 호환용. 이제 IK 초기값으로는 안 씀."""
        q = self.get_joint_positions_pulse()
        return pulse2angle(q)

    def set_joint_positions(self, q, duration):
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration)

    def get_detected_markers(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        if self.image is not None:
            detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
            self.image = None
        else:
            detected_markers = {}
        return detected_markers

    # =====================================================================
    # 1) (선택) 실제 로봇을 q_init 으로 한 번 맞춰두는 함수
    # =====================================================================
    def align_to_init(self, duration=2.0):
        """실제 로봇을 우리가 정한 q_init으로 보냄."""
        self.set_joint_positions(self.q_init, duration)
        time.sleep(duration)
        self.last_q = self.q_init.copy()

    # =====================================================================
    # 2) 카메라에서 본 마커 pose → 월드로 변환 (위치만 신뢰)
    # =====================================================================
    def get_block_pose(self, detection_result: MarkerDetectionResult):
        """
        카메라가 본 마커 pose를 로봇 월드 좌표로 변환해서 4x4로 리턴.
        여기서는 'orientation'은 신뢰 안 하고, grasp() 안에서 덮어쓴다.
        """
        # (원래는 현재 q로 fk(cam) 하던 코드였지만,
        #  여기서는 '카메라가 로봇에 붙어있다'는 세팅만 쓰면 됨.)
        # 그래도 FK는 필요한데, 이 때는 실제 관절을 읽어도 되고,
        # 이미 q_init으로 맞춰뒀으면 q_init으로 FK 해도 된다.

        # 1) 카메라 포즈 (월드 기준) — 여기서는 실제 q를 한 번 읽는 게 맞음
        q_now = self.get_joint_positions()
        T_world_cam = forward_kinematics(q_now, target="cam")

        # 2) 카메라 → 마커 (OpenCV pose)
        R, _ = cv2.Rodrigues(detection_result.rvec)
        t = detection_result.tvec.reshape(3, 1)
        T_cam_marker = np.eye(4, dtype=float)
        T_cam_marker[:3, :3] = R
        T_cam_marker[:3, 3] = t[:, 0]

        # 3) 마커 → 블록 (필요하면 여기서 z 조정)
        T_marker_block = np.eye(4, dtype=float)
        # 예: 마커가 블록 윗면에 붙어 있으면 살짝 내리기
        T_marker_block[2, 3] = -0.01

        T_world_block = T_world_cam @ T_cam_marker @ T_marker_block
        return T_world_block

    # =====================================================================
    # 3) 그랩 시퀀스
    # =====================================================================
    def grasp(self, target_marker_id: int | str) -> bool:
        """
        완전 새 구조:
        - 현재 로봇 상태는 IK에 안 씀
        - 항상 self.q_init 에서 IK 풀어서 보냄
        """
    def grasp_pose(self, T_world_block) -> bool:
        """
        Executes the grasping sequence for a given target pose T_world_block.
        """
        is_success = False

        # (옵션) 시작할 때 실제 로봇도 q_init으로 맞춰두고 싶으면 이거 켜기
        self.align_to_init()

        # 0) 그리퍼 열기
        self.gripper_open(1.0)
        
        R_grasp_top_down = np.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.]], dtype = jnp.float64)
            
        p_target = T_world_block[:3, 3]
        
        # [MODIFIED] Add manual Z-offset (3cm down)
        # Because robot often grasps too high.
        p_target[2] -= 0.03 
        
        T_block= np.eye(4)
        T_block[:3, :3] = R_grasp_top_down
        T_block[:3, 3] = p_target

        # 3) 여기서 orientation을 “집는 자세”로 덮어쓴다
        #    일단은 월드축과 똑같이 (필요하면 살짝 기울이기)
        # original R_pick = np.eye(3)
        #original T_block[:3, :3] = R_pick

        # 4) 접근/리프트 포즈 만들기
        def translate_z(T, dz):
            Tn = T.copy()
            Tn[2, 3] += dz
            return Tn
            

        APPROACH_DZ = 0.05
        LIFT_DZ     = 0.08


        T_app  = translate_z(T_block, +APPROACH_DZ)
        T_lift = translate_z(T_block, +LIFT_DZ)

        # 5) 항상 같은 초기값으로 IK
        q0 = self.get_joint_positions()
        print("\n[IK] q_init used:", q0)

        # [MODIFIED] Two-step logic: Approach & Grasp (Merged), then Lift.
        # We skip the intermediate physical stop at T_app.
        
        # 5-2) 내려오기 (Directly to Grasp Pose)
        # We use q0 as seed.
        sol_grasp = inverse_kinematics(q0, T_block)
        print("[IK] T_block:\n", T_block)
        if not sol_grasp or sol_grasp["sol"] is None:
            print("[GRASP] IK for block failed")
            return False
            
        q_grasp = self._wrap_to_pi(sol_grasp["sol"])
        print("[IK] q_grasp:", q_grasp, "err:", sol_grasp["pos_error"])
        
        pos_error_grasp = sol_grasp["pos_error"]
        if pos_error_grasp < 0.1:
            print("Moving to Grasp Pose...")
            # Slower speed for safety since it's a longer move
            self.set_joint_positions(q_grasp, 2.0) 
            time.sleep(2.0)
        else:
            print("IK solution error too high.")
            return False

        # 집기
        self.gripper_close(0.8)
        time.sleep(0.3)

        # 5-3) 들기
        sol_lift = inverse_kinematics(q_grasp,T_lift)
        print("[IK] T_lift:\n", T_lift)
        print("[IK] sol_lift:", sol_lift)
        if not sol_lift or sol_lift["sol"] is None:
            print("[GRASP] IK for lift failed")
            return False
        q_lift = self._wrap_to_pi(sol_lift["sol"])
        print("[IK] q_lift:", q_lift, "err:", sol_lift["pos_error"])
        pos_error_lift = sol_lift["pos_error"]
        if pos_error_lift < 0.1:
            self.set_joint_positions(q_lift, 1.0)
            time.sleep(1.0)
        else:
            print("IK solution error too high.")
            return False

        # 다음 동작에서 쓸 수 있게 저장만 해둔다
        self.last_q = q_lift.copy()
        is_success = True
        return is_success

    def grasp(self, target_marker_id: int | str) -> bool:
        """
        완전 새 구조:
        - 현재 로봇 상태는 IK에 안 씀
        - 항상 self.q_init 에서 IK 풀어서 보냄
        """
        # 1) 마커 탐색
        detection_result = None
        t0 = time.time()
        while time.time() - t0 < 3.0:
            det = self.get_detected_markers()
            if target_marker_id in det:
                detection_result = det[target_marker_id]
                break
            time.sleep(0.05)
        if detection_result is None:
            print("[GRASP] marker not found")
            return False

        # 2) 마커 → 월드 블록 포즈
        T_block_from_cam = self.get_block_pose(detection_result)
        
        return self.grasp_pose(T_block_from_cam)

    # =====================================================================
    # 4) place 도 q_init 기반으로
    # =====================================================================
    def place(self, action_name) -> bool:
        self.align_to_init()
        APPROACH_DZ = 0.06
        LIFT_DZ     = 0.10
        DROP_ZOFF   = 0.02

        place_targets_xyz = {
            "blue_3":  np.array([0.30,  0.00, 0.25]),
            "green_3": np.array([0.30,  0.06, 0.25]),
            "red_3":   np.array([0.30, -0.06, 0.25]),
        }
        if action_name not in place_targets_xyz:
            print(f"[WARN] Unknown place action_name: {action_name}")
            return False

        p = place_targets_xyz[action_name]
        R_down = np.array([
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.]], dtype = jnp.float64)
        z_axis = np.array([0.0, 0.0, 1.0])

        T_place = np.eye(4); T_place[:3, :3] = R_down; T_place[:3, 3] = p
        T_above = T_place.copy(); T_above[:3, 3] += z_axis * APPROACH_DZ
        T_drop  = T_place.copy(); T_drop [:3, 3] += z_axis * DROP_ZOFF
        T_up    = T_place.copy(); T_up   [:3, 3] += z_axis * (LIFT_DZ + DROP_ZOFF)

        q0 = self.q_init.copy()

        # 1) 위로
        sol1 = inverse_kinematics(q0, T_above)
        if not sol1 or sol1["sol"] is None:
            print("[PLACE] IK above fail")
            return False
        q_above = self._wrap_to_pi(sol1["sol"])
        self.set_joint_positions(q_above, 1.3); time.sleep(1.3)

        # 2) 내려가기
        sol2 = inverse_kinematics(q0, T_drop)
        if not sol2 or sol2["sol"] is None:
            print("[PLACE] IK drop fail")
            return False
        q_drop = self._wrap_to_pi(sol2["sol"])
        self.set_joint_positions(q_drop, 1.1); time.sleep(1.1)

        # 3) 열기
        self.gripper_open(0.7); time.sleep(0.3)

        # 4) 위로 빼기
        sol3 = inverse_kinematics(q0, T_up)
        if not sol3 or sol3["sol"] is None:
            print("[PLACE] IK up fail")
            return False
        q_up = self._wrap_to_pi(sol3["sol"])
        self.set_joint_positions(q_up, 1.1); time.sleep(1.1)

        self.last_q = q_up.copy()
        return True

    # =====================================================================
    # 5) 헬퍼
    # =====================================================================
    def _wrap_to_pi(self, q):
        q = np.asarray(q, dtype=float)
        return q
