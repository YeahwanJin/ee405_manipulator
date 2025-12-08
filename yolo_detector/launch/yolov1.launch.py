# yolo_detector/launch/yolov1_with_depth_cam.launch.py
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    peripherals_share = get_package_share_directory('peripherals')

    env_need_compile = SetEnvironmentVariable('need_compile', 'False')
    
    env_camera_type  = SetEnvironmentVariable('DEPTH_CAMERA_TYPE', os.environ.get('DEPTH_CAMERA_TYPE', 'USB'))

    depth_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(peripherals_share, 'launch', 'depth_camera.launch.py'))
    )
    
    yolov1 = Node(
        package='yolo_detector',
        executable='yolov1_inference',
        name='yolov1_inference',
        output='screen'
    )

    return LaunchDescription([
        env_need_compile,
        env_camera_type,
        depth_camera,
        yolov1
    ])
