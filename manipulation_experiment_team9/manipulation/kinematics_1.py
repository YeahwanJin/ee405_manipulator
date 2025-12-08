import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import least_squares
from .utils.kinematics_utils import get_urdf
from jaxlie import SO3

#####

#####
def hat(w):
    wx, wy, wz = w
    return jnp.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def rot_axis_angle(w, theta):
    w = jnp.asarray(w, dtype=float)
    n = jnp.linalg.norm(w)
    
    # 🚨 수정: if n == 0: 문을 jnp.where로 대체합니다.
    
    # n이 0이 아닐 때의 로직 (회전 행렬 계산)
    w_normalized = w / n
    w_hat = hat(w_normalized)
    R_nonzero = jnp.eye(3) + jnp.sin(theta)*w_hat + (1-jnp.cos(theta))*(w_hat @ w_hat)
    
    # n이 0일 때의 로직 (단위 행렬)
    R_zero = jnp.eye(3)
    
    # n이 0인지 여부에 따라 두 결과 중 하나를 선택합니다.
    # jnp.allclose는 부동소수점 비교에 더 안전합니다.
    is_zero = jnp.allclose(n, 0.0) 
    
    # is_zero가 True이면 R_zero, False이면 R_nonzero를 반환합니다.
    return jnp.where(is_zero, R_zero, R_nonzero)

    
def make_T(R, p):
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3,  3].set(p)

    return T

def forward_kinematics(
    q: jnp.ndarray, 
    urdf: dict,
    target: str,
) -> jnp.ndarray:
    """
    Compute the forward kinematics for the robotic arm.
    Args:
        q (np.ndarray): Joint angles (1D array of size 5) in radian.
        target (str): Target frame to compute the pose for. Options are:
                      'j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp'.
                      Default is 'j1'.
    Returns:
        jnp.ndarray: 4x4 transformation matrix of the target frame.
    """
    urdf_dict = urdf
    T_base = jnp.asarray(urdf_dict["T_base"])
    T_base_joint_1 = jnp.asarray(urdf_dict["T_base_joint_1"])
    T_joint1_joint_2 = jnp.asarray(urdf_dict["T_joint1_joint_2"])
    T_joint2_joint_3 = jnp.asarray(urdf_dict["T_joint2_joint_3"])
    T_joint3_joint_4 = jnp.asarray(urdf_dict["T_joint3_joint_4"])
    T_joint4_joint_5 = jnp.asarray(urdf_dict["T_joint4_joint_5"])
    T_joint5_TCP_offset = jnp.asarray(urdf_dict["T_joint5_TCP_offset"])
    T_joint4_cam_offset = jnp.asarray(urdf_dict["T_joint4_cam_offset"])

    j1_axis = jnp.asarray(urdf_dict["Axis_joint_1"])
    j2_axis = jnp.asarray(urdf_dict["Axis_joint_2"])
    j3_axis = jnp.asarray(urdf_dict["Axis_joint_3"])
    j4_axis = jnp.asarray(urdf_dict["Axis_joint_4"])
    j5_axis = jnp.asarray(urdf_dict["Axis_joint_5"])

    T = dict()
    T["j1"] = T_base @ T_base_joint_1 @ make_T(rot_axis_angle(j1_axis, q[0]), jnp.zeros(3))
    T["j2"] = T["j1"] @ T_joint1_joint_2 @ make_T(rot_axis_angle(j2_axis, q[1]), jnp.zeros(3))
    T["j3"] = T["j2"] @ T_joint2_joint_3 @ make_T(rot_axis_angle(j3_axis, q[2]), jnp.zeros(3))
    T["j4"] = T["j3"] @ T_joint3_joint_4 @ make_T(rot_axis_angle(j4_axis, q[3]), jnp.zeros(3))
    T["j5"] = T["j4"] @ T_joint4_joint_5 @ make_T(rot_axis_angle(j5_axis, q[4]), jnp.zeros(3))
    T["tcp"] = T["j5"] @ T_joint5_TCP_offset
    T["cam"] = T["j4"] @ T_joint4_cam_offset
    ######
    ## TODO : Implement Forward Kinematics using the robot's URDF parameters.
    ## You can refer to the URDF parameters defined in get_urdf() function in utils/kinematics_utils.py.
    pose = T[target];breakpoint()

    #####

    return pose


# inverse_kinematics가 데이터 준비를 담당합니다.
def inverse_kinematics(
    q: np.ndarray, 
    T_target: np.ndarray
) -> dict | None:

    # 1. 데이터 준비: JIT 컴파일 전에 모든 외부 데이터를 불러오고 JAX 배열로 변환
    urdf = get_urdf();print("11111111111111")
    urdf_jax = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in urdf.items()};print("22222222222222222")
    T_target_jax = jnp.asarray(T_target);print("33333333333333333333333")

    # 2. JIT 컴파일할 순수 계산 함수 정의
    @jax.jit
    def ik_loss(q_current):
        # 준비된 데이터를 사용하여 계산만 수행
        T_current = forward_kinematics(q_current, urdf_jax, target='tcp')

        pos_error = T_current[:3, 3] - T_target_jax[:3, 3]

        R_current_matrix = T_current[:3, :3]
        R_target_matrix = T_target_jax[:3, :3]
        R_error_matrix = R_current_matrix.T @ R_target_matrix 
        ori_error = SO3.from_matrix(R_error_matrix).log()

        return jnp.concatenate([pos_error, ori_error])

    # 3. 최적화 실행
    jacobian_fn = jax.jit(jax.jacfwd(ik_loss));print("555555555555555555555555555555")

    ls_result = least_squares(
        fun=ik_loss, 
        x0=q,             
        jac=jacobian_fn,  
        method='lm',
        ftol=1e-6,
        xtol=1e-6
    )
    
    if not ls_result.success:
        return None

    ik_solution = ls_result.x
    
    # 최종 에러 계산 시에도 np.asarray로 변환하여 사용
    final_pose_np = np.asarray(forward_kinematics(ik_solution, urdf_jax, target='tcp'))
    pos_error = float(np.linalg.norm(final_pose_np[:3, 3] - T_target[:3, 3]))

    res = {
        "sol": np.asarray(ik_solution),
        "pos_error": pos_error
    }

    return res
