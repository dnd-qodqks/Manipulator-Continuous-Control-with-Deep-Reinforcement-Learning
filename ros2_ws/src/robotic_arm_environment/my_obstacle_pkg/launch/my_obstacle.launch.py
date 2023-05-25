import os
import launch
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_dir = get_package_share_directory('my_obstacle_pkg') 
    urdf_path = os.path.join(pkg_dir, 'description', 'obstacle.urdf')
    
    spawn_entity_obstacle = Node(package='gazebo_ros',
                              executable='spawn_entity.py',
                              arguments=['-entity', 'obstacle', '-file', urdf_path],
                              output='screen')
    
    return launch.LaunchDescription([spawn_entity_obstacle])