import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import least_squares
from .utils.kinematics_utils import get_urdf
from jaxlie import SO3

#####

def forward_kinematics(
    q: np.ndarray, 
    target: str,
) -> np.ndarray:
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

    ######
    ## TODO : Implement Forward Kinematics using the robot's URDF parameters.
    ## You can refer to the URDF parameters defined in get_urdf() function in utils/kinematics_utils.py.
    pose = None
     # ---- Minimal FK with JAX + jaxlie (uses URDF data) ----
        # ---- FK (concise) ----
    U = get_urdf()
    Tb   = jnp.array(U["T_base"], dtype=jnp.float32)
    Tb1  = jnp.array(U["T_base_joint_1"], dtype=jnp.float32)
    T12  = jnp.array(U["T_joint1_joint_2"], dtype=jnp.float32)
    T23  = jnp.array(U["T_joint2_joint_3"], dtype=jnp.float32)
    T34  = jnp.array(U["T_joint3_joint_4"], dtype=jnp.float32)
    T45  = jnp.array(U["T_joint4_joint_5"], dtype=jnp.float32)
    T5t  = jnp.array(U["T_joint5_TCP_offset"], dtype=jnp.float32)
    T4c  = jnp.array(U["T_joint4_cam_offset"], dtype=jnp.float32)
    a1,a2,a3,a4,a5 = [jnp.array(U[f"Axis_joint_{i}"], dtype=jnp.float32) for i in range(1,6)]
    R = lambda a,t: SO3.exp(a / (jnp.linalg.norm(a)+1e-12) * t).as_matrix()
    SE3 = lambda Rm: jnp.vstack([jnp.hstack([Rm, jnp.zeros((3,1),jnp.float32)]), jnp.array([0,0,0,1],jnp.float32)])

    q = jnp.asarray(q, dtype=jnp.float32)
    T = Tb @ Tb1 @ SE3(R(a1,q[0]));      P = {"j1":T}
    T = T @ T12 @ SE3(R(a2,q[1]));       P["j2"]=T
    T = T @ T23 @ SE3(R(a3,q[2]));       P["j3"]=T
    T = T @ T34 @ SE3(R(a4,q[3]));       P["j4"]=T;  P["cam"]=T @ T4c
    T = T @ T45 @ SE3(R(a5,q[4]));       P["j5"]=T;  P["tcp"]=T @ T5t

    pose = np.asarray(P.get(target, P["j5"]), dtype=float)

    return pose

def inverse_kinematics(
    q: np.ndarray, 
    T_target: np.ndarray
) -> dict | None:
    """
    Compute the inverse kinematics for the robotic arm.
    Args:
        q (np.ndarray): Initial joint angles (1D array of size 5) in radian.
        T_target (np.ndarray): 4x4 transformation matrix of the target end-effector pose (tcp).
    
        Returns:
            dict | None: A dictionary containing the solution joint angles and position error, or None if failed.
                {
                    "sol": np.ndarray,  # Solution joint angles (1D array of size 5) in radian or None if failed
                    "pos_error": float  # Position error in meters or None if failed
                }
    """

    #####
    ## TODO : Implement Inverse Kinematics using numerical optimization.
    ## You may use jax.jacfwd, scipy.optimize.least_squares and forward_kinematics function for this purpose.
    ## When the optimization fails, return None.
    ## (Optional) Refer to jax.jit function to speed up the repeated computation.

    pos_error : float = None
    ik_solution : np.ndarray = None
    U = get_urdf()
    Tb   = jnp.array(U["T_base"], dtype=jnp.float32)
    Tb1  = jnp.array(U["T_base_joint_1"], dtype=jnp.float32)
    T12  = jnp.array(U["T_joint1_joint_2"], dtype=jnp.float32)
    T23  = jnp.array(U["T_joint2_joint_3"], dtype=jnp.float32)
    T34  = jnp.array(U["T_joint3_joint_4"], dtype=jnp.float32)
    T45  = jnp.array(U["T_joint4_joint_5"], dtype=jnp.float32)
    T5t  = jnp.array(U["T_joint5_TCP_offset"], dtype=jnp.float32)
    a1,a2,a3,a4,a5 = [jnp.array(U[f"Axis_joint_{i}"], dtype=jnp.float32) for i in range(1,6)]
    R = lambda a,t: SO3.exp(a / (jnp.linalg.norm(a)+1e-12) * t).as_matrix()
    SE3 = lambda Rm: jnp.vstack([jnp.hstack([Rm, jnp.zeros((3,1),jnp.float32)]), jnp.array([0,0,0,1],jnp.float32)])
    T_goal = jnp.asarray(T_target, dtype=jnp.float32)

    def fk_tcp(qv):
        T = Tb @ Tb1 @ SE3(R(a1,qv[0]))
        T = T @ T12 @ SE3(R(a2,qv[1]))
        T = T @ T23 @ SE3(R(a3,qv[2]))
        T = T @ T34 @ SE3(R(a4,qv[3]))
        T = T @ T45 @ SE3(R(a5,qv[4]))
        return T @ T5t

    def err(qv):
        Te = fk_tcp(qv); Re, pe = Te[:3,:3], Te[:3,3]
        Rt, pt = T_goal[:3,:3], T_goal[:3,3]
        Rerr = Re.T @ Rt
        rot  = SO3.from_matrix(Rerr).log()
        pos  = pt - pe
        return jnp.concatenate([pos, rot], 0)

    jac = jax.jacfwd(err)
    fun = lambda x: np.asarray(err(jnp.asarray(x, jnp.float32)), dtype=float)
    jacf= lambda x: np.asarray(jac(jnp.asarray(x, jnp.float32)), dtype=float)
    
    joint_limits_min = (-np.pi , -np.pi, -np.pi, -np.pi, -np.pi)
    joint_limits_max = (np.pi , np.pi, np.pi, np.pi, np.pi)

    sol = least_squares(fun, x0=np.asarray(q, float), jac=jacf, method="trf",
                        xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=200, bounds=(joint_limits_min, joint_limits_max))

    if not sol.success:
        return None

    ik_solution = sol.x.astype(float)
    
    # --- [추가된 코드 시작] ---
    
    # 최종 6D 에러 벡터 [pos, rot]를 계산합니다.
    final_error_vector = fun(ik_solution)
    
    # 6D 벡터를 3D 위치 에러와 3D 회전 에러로 분리합니다.
    final_pos_error_vec = final_error_vector[:3]
    final_rot_error_vec = final_error_vector[3:]
    
    # 요청하신 에러 값들을 출력합니다.
    print("-" * 30)
    print(f"calculated fk value: {fk_tcp(ik_solution)}")
    print(f"Final IK Solution Error:")
    print(f"  Position Error Vector (pos): {final_pos_error_vec}")
    print(f"  Rotation Error Vector (rot): {final_rot_error_vec}")
    print("-" * 30)
    
    # --- [추가된 코드 끝] ---
    
    pos_error = float(np.linalg.norm(fun(ik_solution)[:3]))

    #####

    result = {}
    result["sol"] = ik_solution
    result["pos_error"] = pos_error

    return result
