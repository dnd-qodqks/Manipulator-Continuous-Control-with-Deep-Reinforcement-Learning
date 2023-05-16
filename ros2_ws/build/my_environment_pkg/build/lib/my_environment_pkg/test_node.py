import rclpy
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def main():
    rclpy.init()
    # my_sphere_files       = get_package_share_directory('my_sphere_pkg')
    # print("my_sphere_files:", my_sphere_files)
    
    my_doosan_robot_files = get_package_share_directory('my_doosan_pkg')
    print("my_doosan_robot_files:", my_doosan_robot_files)

    # my_environmets_files  = get_package_share_directory('my_environment_pkg')
    # print("my_environmets_files:", my_environmets_files)

    doosan_robot = IncludeLaunchDescription(PythonLaunchDescriptionSource(my_doosan_robot_files + '/launch/my_doosan_controller.launch.py')) 
    print("PythonLaunchDescriptionSource: ", PythonLaunchDescriptionSource(my_doosan_robot_files + '/launch/my_doosan_controller.launch.py'))
    print("doosan_robot:", doosan_robot)
    
    rclpy.shutdown()
    
if __name__ == '__main__':
	main()