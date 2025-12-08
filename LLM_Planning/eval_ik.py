from manipulation.utils.test_fk_ik import test_custom_ik_accuracy
from manipulation.kinematics import inverse_kinematics
    
if __name__ == "__main__":
    test_custom_ik_accuracy(inverse_kinematics)
