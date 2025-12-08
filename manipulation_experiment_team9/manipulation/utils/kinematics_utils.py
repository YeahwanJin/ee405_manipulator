from math import degrees, radians, atan2, asin, sqrt
import numpy as np

def get_urdf():
    """
    Generates and returns a dictionary containing the kinematic parameters of the robot.

    NOTE:
         All transformation matrices are defined based on the robot's 'zero position'
        (when all joint angles are zero). Those values are parsed from the URDF file.
         The dictionary includes both the transformation matrices and the rotation axes for each joint.

    Returns:
        dict: A dictionary containing the kinematic parameters of the robot.
              The keys and their meanings are as follows:
            - "T_base" (np.ndarray):
                A 4x4 homogeneous transformation matrix from the world frame to the
                robot's base frame.
            - "T_base_joint_1" (np.ndarray):
                The static transform from the base frame to the joint 1 frame.
            - "T_joint1_joint_2" (np.ndarray):
                The static transform from the joint 1 frame to the joint 2 frame.
            - "T_joint2_joint_3" (np.ndarray):
                The static transform from the joint 2 frame to the joint 3 frame.
            - "T_joint3_joint_4" (np.ndarray):
                The static transform from the joint 3 frame to the joint 4 frame.
            - "T_joint4_joint_5" (np.ndarray):
                The static transform from the joint 4 frame to the joint 5 frame.
            - "T_joint5_TCP_offset" (np.ndarray):
                The static transform from the joint 5 (last joint) frame to the
                Tool Center Point (TCP) frame. This defines the robot's tool.
            - "T_joint4_cam_offset" (np.ndarray):
                The static transform from the joint 4 frame to the camera frame,
                representing the camera's mounting position on the robot arm.
            - "Axis_joint_1" (np.ndarray): A 3D vector representing the rotation axis of joint 1.
            - "Axis_joint_2" (np.ndarray): A 3D vector representing the rotation axis of joint 2.
            - "Axis_joint_3" (np.ndarray): A 3D vector representing the rotation axis of joint 3.
            - "Axis_joint_4" (np.ndarray): A 3D vector representing the rotation axis of joint 4.
            - "Axis_joint_5" (np.ndarray): A 3D vector representing the rotation axis of joint 5.
    """
    ### --- kinematics ---
    T_base = np.array([[1.        , 0.        , 0.          , 0.],
                        [0.        , 1.        , 0.          , 0.],
                        [0.        , 0.        , 1.          , 0.11609108215767461],
                        [0.        , 0.        , 0.          , 1.]])

    T_joint1 = np.array([[1.        , 0.        , 0.        , 0.0251328065010765],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.0774026880954513],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint2 = np.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.0338648012164686],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint3 = np.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.129416446394797],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint4 = np.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.129444631186569],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint5 = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.0544833339503674],
                        [0., 0., 0., 1.]])
    ##end-effector frame
    T_TCP_offset = np.array([[1.        , 0.        , 0.        , 0.],
                            [0.        , 1.        , 0.        , 0.],
                            [0.        , 0.        , 1.        , 0.113],  # 0.113 (Measured with closed gripper)
                            [0.        , 0.        , 0.        , 1.]])

    T_cam_offset = np.array([[0.        , 1.        , 0.        , -0.0507060266977644],
                            [-1.       , 0.        , 0.        , 0.],
                            [0.        , 0.        , 1.        , 0.065013484],
                            [0.        , 0.        , 0.        , 1.]])
    
    j1_axis = np.array([0, 0, -1])
    j2_axis = np.array([0, 1, 0])
    j3_axis = np.array([0, 1, 0])
    j4_axis = np.array([0, 1, 0])
    j5_axis = np.array([0, 0, -1])
    
    urdf_dict = {
        "T_base": T_base,
        "T_base_joint_1": T_joint1,
        "T_joint1_joint_2": T_joint2,
        "T_joint2_joint_3": T_joint3,
        "T_joint3_joint_4": T_joint4,
        "T_joint4_joint_5": T_joint5,
        "T_joint5_TCP_offset": T_TCP_offset,
        "T_joint4_cam_offset": T_cam_offset,
        "Axis_joint_1": j1_axis,
        "Axis_joint_2": j2_axis,
        "Axis_joint_3": j3_axis,
        "Axis_joint_4": j4_axis,
        "Axis_joint_5": j5_axis,
    }

    return urdf_dict
                            
joint1_map = [0, 1000, 500, 120, -120, 0]
joint2_map = [0, 1000, 500, 120, -120, 0]
joint3_map = [0, 1000, 500, 120, -120, 0]
joint4_map = [0, 1000, 500, 120, -120, 0]
joint5_map = [0, 1000, 500, 120, -120, 0]


def angle_transform(angle, param, inverse=False):
    if inverse:
        new_angle = ((angle - param[5]) / (param[4] - param[3])) * (param[1] - param[0]) + param[2]
    else:
        new_angle = ((angle - param[2]) / (param[1] - param[0])) * (param[4] - param[3]) + param[5]

    return new_angle

def pulse2angle(pulse):
    theta1 = angle_transform(pulse[0], joint1_map)
    theta2 = angle_transform(pulse[1], joint2_map)
    theta3 = angle_transform(pulse[2], joint3_map)
    theta4 = angle_transform(pulse[3], joint4_map)
    theta5 = angle_transform(pulse[4], joint5_map)
    
    #print(theta1, theta2, theta3, theta4, theta5)
    q = radians(theta1), radians(theta2), radians(theta3), radians(theta4), radians(theta5)
    return np.array(q)

def angle2pulse(angle):
    pulse = []
    theta1 = int(angle_transform(degrees(angle[0]), joint1_map, True))
    theta2 = int(angle_transform(degrees(angle[1]), joint2_map, True))
    theta3 = int(angle_transform(degrees(angle[2]), joint3_map, True))
    theta4 = int(angle_transform(degrees(angle[3]), joint4_map, True))
    theta5 = int(angle_transform(degrees(angle[4]), joint5_map, True))

    pulse = np.array([theta1, theta2, theta3, theta4, theta5])

    return pulse

def angle2pulse_single(angle, joint_idx):
    if joint_idx == 0:
        pulse = int(angle_transform(degrees(angle), joint1_map, True))
    elif joint_idx == 1:
        pulse = int(angle_transform(degrees(angle), joint2_map, True))
    elif joint_idx == 2:
        pulse = int(angle_transform(degrees(angle), joint3_map, True))
    elif joint_idx == 3:
        pulse = int(angle_transform(degrees(angle), joint4_map, True))
    elif joint_idx == 4:
        pulse = int(angle_transform(degrees(angle), joint5_map, True))
    else:
        raise ValueError("Invalid joint index")

    return pulse


def check_se3(T):
    assert T.shape == (4,4), "T must be 4x4 matrix"
    assert np.allclose(T[3,:], np.array([0.,0.,0.,1.])), "Last row must be [0,0,0,1]"
    assert np.allclose(T[:3,:3] @ T[:3,:3].T, np.eye(3)), "Rotation part must be orthogonal"
    assert np.isclose(np.linalg.det(T[:3,:3]), 1.0), "Rotation part must have determinant 1"
