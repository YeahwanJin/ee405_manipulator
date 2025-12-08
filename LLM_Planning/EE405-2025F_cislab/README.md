# EE405-2025F: Path Planning and Navigation

This repository contains ROS2 packages and configuration files for the EE405-2025F course, focusing on path planning, navigation, and visualization.

## Repository Structure

```
packages/
├── custom_visualize/              # ROS2 visualization package
│   ├── package.xml                # Package metadata and dependencies
│   ├── setup.py                   # Python package setup configuration
│   ├── setup.cfg                  # Python package setup configuration
│   ├── resource/
│   │   └── custom_visualize       # Resource marker file
│   ├── custom_visualize/          # Main Python package
│   │   ├── __init__.py           # Package initialization
│   │   └── path_visualizer.py    # Main visualization node
│   └── test/                      # Unit tests
│       ├── test_copyright.py      # Copyright compliance test
│       ├── test_flake8.py        # Code style test
│       └── test_pep257.py        # Docstring style test
├── nav2_controller_mppi.yaml      # MPPI controller configuration
└── README.md                      # This documentation file
```

## Contents

### 1. Custom Visualization Package (`custom_visualize/`)

A ROS2 Python package for visualizing robot path planning and navigation:

- **Package Name**: `custom_visualize`
- **Maintainer**: Bon Choe (bonjae.choe+TA@gmail.com)
- **Description**: A package for visualizing global and local path planning

#### Features

- **Path Visualization**: Displays planned global paths in RED
- **Trajectory Visualization**: Shows local planned trajectories in BLUE (temporary, 1-second lifetime)
- **Executed Path Tracking**: Records and displays the robot's actual executed path in GRAY
- **Real-time Updates**: Subscribes to odometry data for continuous path tracking

#### ROS2 Topics

- **Subscriptions**:
  - `/plan` (nav_msgs/Path) - Global planned path
  - `/local_plan` (nav_msgs/Path) - Local planned trajectory
  - `/odom` (nav_msgs/Odometry) - Robot odometry data
- **Publications**:
  - `/visualization_marker` (visualization_msgs/Marker) - Path markers for RViz
  - `/executed_path` (nav_msgs/Path) - Accumulated executed path

#### Installation

1. Clone this repository into your ROS2 workspace
2. Build the package:
   ```bash
   colcon build --packages-select custom_visualize
   ```
3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

#### Usage

Run below command after launching your ROS2 navigation stack (see lab document for details):

```bash
ros2 run custom_visualize path_visualizer
```

### 2. Navigation Configuration (`nav2_controller_mppi.yaml`)

This is a standalone configuration file for the Nav2 MPPI (Model Predictive Path Integral) controller. It is not part of the `custom_visualize` package but is provided as a reference configuration for the course.

#### Usage

Please move this file to your Nav2 configuration directory as explained in the lab document.

## Dependencies

- ROS2 (tested with appropriate ROS2 distribution)
- Nav2 navigation stack
- Standard ROS2 message packages:
  - `nav_msgs`
  - `geometry_msgs`
  - `visualization_msgs`
  - `builtin_interfaces`
