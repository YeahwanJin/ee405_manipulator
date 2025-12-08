from manipulation.utils.test_fk_ik import test_custom_fk_accuracy
from manipulation.kinematics import forward_kinematics
    
if __name__ == "__main__":
    test_custom_fk_accuracy(forward_kinematics)
