import numpy as np
from pathlib import Path
from functools import partial
import yourdfpy

URDF_PATH = Path("resources/jetrover_description/urdf/jetrover.urdf")

def calculate_fk_accuracy(T_predicted_list, T_ground_truth_list):
    trans_errors = []
    rot_errors = []

    for T_pred, T_gt in zip(T_predicted_list, T_ground_truth_list):
        p_pred, p_gt = T_pred[:3, 3], T_gt[:3, 3]
        trans_errors.append(np.linalg.norm(p_pred - p_gt))

        R_pred, R_gt = T_pred[:3, :3], T_gt[:3, :3]
        R_err = R_pred @ R_gt.T
        
        trace_val = np.trace(R_err)
        angle_rad = np.arccos(np.clip((trace_val - 1) / 2.0, -1.0, 1.0))
        rot_errors.append(angle_rad)

    return np.array(trans_errors), np.array(rot_errors)


def test_custom_fk_accuracy(fk_func, num_tests=100, seed=42):
    # Load URDF
    robot_urdf = yourdfpy.URDF.load(
        URDF_PATH,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=URDF_PATH.parent)
    )

    # Extract joint limits manually

    # Prepare limit arrays
    lower_limits = [-2.09, -2.09, -2.09, -2.09, -2.09]
    upper_limits = [2.09, 2.09, 2.09, 2.09, 2.09]
    
    lower_limits = np.array(lower_limits)
    upper_limits = np.array(upper_limits)

    # Custom FK targets vs Ground Truth links (as per your comment)
    link_names_custom = ['j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp']
    frames_to_visualize = ['link1', 'link2', 'link3', 'link4', 'link5', 'depth_cam_frame', 'end_effector_link']

    np.random.seed(seed)
    all_trans_errors = []
    all_rot_errors = []

    for i in range(num_tests):
        q = np.random.uniform(lower_limits, upper_limits)

        fk_custom_list = []
        fk_gt_list = []

        try:
            q = np.concatenate([q, np.array([0])])
            robot_urdf.update_cfg(q)
        except Exception as e:
            print(f"[{i}] Skipping config due to yourdfpy update error: {e}")
            continue

        for link_c, link_gt in zip(link_names_custom, frames_to_visualize):
            try:
                T_custom = np.array(fk_func(q, target=link_c))
                T_gt = robot_urdf.get_transform(link_gt)
                fk_custom_list.append(T_custom)
                fk_gt_list.append(T_gt)
            except Exception as e:
                print(f"[{i}] FK Error on link {link_c}/{link_gt}: {e}")
                break

        if len(fk_custom_list) != len(link_names_custom):
            continue  # Skip incomplete sample

        trans_errs, rot_errs = calculate_fk_accuracy(fk_custom_list, fk_gt_list)
        all_trans_errors.append(trans_errs)
        all_rot_errors.append(rot_errs)

    all_trans_errors = np.array(all_trans_errors)
    all_rot_errors = np.array(all_rot_errors)

    print(f"\n--- FK Accuracy Evaluation over {len(all_trans_errors)} valid samples ---")
    print(f"Mean Translation Error: {np.mean(all_trans_errors):.3f} m")
    print(f"Max Translation Error: {np.max(all_trans_errors):.3f} m")
    print(f"Mean Rotation Error:    {np.mean(all_rot_errors):.3f} rad")
    print(f"Max Rotation Error:     {np.max(all_rot_errors):.3f} rad")
    print("-------------------------------------------------------------")

    return all_trans_errors, all_rot_errors
    
def test_custom_ik_accuracy(ik_func, num_tests=100, seed=42, tol=1e-3):
    from functools import partial
    import numpy as np
    import yourdfpy

    # Load URDF
    robot_urdf = yourdfpy.URDF.load(
        URDF_PATH,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=URDF_PATH.parent)
    )

    # Define joint limits
    lower_limits = np.array([-2.09, -2.09, -2.09, -2.09, -2.09])
    upper_limits = np.array([2.09, 2.09, 2.09, 2.09, 2.09])

    np.random.seed(seed)
    ik_errors = []
    ik_success_count = 0
    trans_errors = []
    rot_errors = []

    for i in range(num_tests):
        q_rand = np.random.uniform(lower_limits, upper_limits)
        q_full = np.concatenate([q_rand, [0]])  # Add dummy for fixed joint if needed

        try:
            # Ground truth FK pose from sampled q_rand
            robot_urdf.update_cfg(q_full)
            T_target = robot_urdf.get_transform("end_effector_link")

            # Run custom FK on IK result
            q_ik = ik_func(np.zeros(5), T_target)
            if q_ik is not None:
                q_error = np.linalg.norm(q_rand[:5] - q_ik["sol"])
                
                # Forward kinematics
                q_full = np.concatenate([q_ik["sol"], [0]])  
                robot_urdf.update_cfg(q_full)
                T_target_hat = robot_urdf.get_transform("end_effector_link")
                
                # Compute FK error 
                trans_errs, rot_errs = calculate_fk_accuracy([T_target_hat], [T_target])
                
                trans_errors.append(trans_errs)
                rot_errors.append(rot_errs)
                
                ik_success_count += 1
                ik_errors.append(q_error)

        except Exception as e:
            print(f"[{i}] Error during IK testing: {e}")
            continue
            
        trans_error_mean = np.mean(trans_errors)
        rot_error_mean = np.mean(rot_errors)

    print(f"\n {ik_success_count} / {num_tests} IK solutions found : ")
    print(f"    Mean Translation Error: {trans_error_mean:.3f} m")
    print(f"    Mean Rotation Error: {rot_error_mean:.3f} rad")
    print("-------------------------------------------------------------")

    return ik_errors
    
if __name__ == "__main__":
    print("start")
    from manipulation.kinematics import forward_kinematics
    # test_custom_fk_accuracy(forward_kinematics)
    
    from manipulation.kinematics import inverse_kinematics
    test_custom_ik_accuracy(inverse_kinematics, num_tests=10)

