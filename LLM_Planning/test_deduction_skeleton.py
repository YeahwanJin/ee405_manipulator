
import sys
import os
import unittest
from unittest.mock import MagicMock

# Mock rclpy before importing final_project
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.action'] = MagicMock()
sys.modules['nav2_msgs.action'] = MagicMock()
sys.modules['sensor_msgs.msg'] = MagicMock()
sys.modules['cv_bridge'] = MagicMock()
sys.modules['cv2'] = MagicMock()

# Import the module
sys.path.append('/home/ubuntu/ros2_ws/src/LLM_Planning')
import final_project

class TestDeduction(unittest.TestCase):
    def test_deduction_red_empty_unknown(self):
        # Scenario: 
        # Loc 1: Red
        # Loc 2: Empty Spot
        # Loc 3: None (Unknown)
        # Active: Red, Blue
        # Expectation: Loc 3 -> Blue
        
        # Setup
        nav_mock = MagicMock()
        mission = final_project.SmartMission(nav_mock)
        
        # Override map
        mission.world_map = {
            "loc_1": {"coords": (1.0, 1.0), "zone_id": None, "cube_color": "red"},
            "loc_2": {"coords": (2.0, 2.0), "zone_id": None, "cube_color": "empty_spot"},
            "loc_3": {"coords": (3.0, 3.0), "zone_id": None, "cube_color": None}
        }
        
        print("\n[Test] Running Deduction Test: Red, Empty, Unknown -> Blue")
        mission.perform_deduction(active_colors=["red", "blue"])
        
        # Assertions
        loc3_color = mission.world_map["loc_3"]["cube_color"]
        self.assertEqual(loc3_color, "blue", "Failed to deduce Loc 3 is Blue")

    def test_deduction_zone_id(self):
        # Scenario:
        # Loc 1: Zone 1
        # Loc 2: Zone 2
        # Loc 3: None (Unknown Zone)
        # Expectation: Loc 3 -> Zone 3
        nav_mock = MagicMock()
        mission = final_project.SmartMission(nav_mock)
        
        mission.world_map = {
            "loc_1": {"coords": (1.0, 1.0), "zone_id": 1, "cube_color": "red"},
            "loc_2": {"coords": (2.0, 2.0), "zone_id": 2, "cube_color": "blue"},
            "loc_3": {"coords": (3.0, 3.0), "zone_id": None, "cube_color": None}
        }
        
        print("\n[Test] Running Deduction Test: Zone 1, Zone 2, Unknown -> Zone 3")
        mission.perform_deduction(active_colors=["red", "blue"])
        
        loc3_zone = mission.world_map["loc_3"]["zone_id"]
        self.assertEqual(loc3_zone, 3, "Failed to deduce Loc 3 is Zone 3")

    def test_deduction_no_deduction_if_ambiguous(self):
        # Scenario:
        # Loc 1: Red
        # Loc 2: None
        # Loc 3: None
        # Active: Red, Blue
        # Expectation: Loc 2, Loc 3 remain None (Deduction fail)
        nav_mock = MagicMock()
        mission = final_project.SmartMission(nav_mock)
        
        mission.world_map = {
            "loc_1": {"coords": (1.0, 1.0), "zone_id": None, "cube_color": "red"},
            "loc_2": {"coords": (2.0, 2.0), "zone_id": None, "cube_color": None},
            "loc_3": {"coords": (3.0, 3.0), "zone_id": None, "cube_color": None}
        }
        
        print("\n[Test] Running Deduction Test: Ambiguous -> No Change")
        mission.perform_deduction(active_colors=["red", "blue"])
        
        self.assertIsNone(mission.world_map["loc_2"]["cube_color"])
        self.assertIsNone(mission.world_map["loc_3"]["cube_color"])

if __name__ == '__main__':
    unittest.main()
