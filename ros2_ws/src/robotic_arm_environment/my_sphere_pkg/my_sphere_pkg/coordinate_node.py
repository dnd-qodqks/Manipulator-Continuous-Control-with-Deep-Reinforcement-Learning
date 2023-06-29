
'''

Author: David Valencia
Date: 11 / 08 /2021

Describer: 

		   This script does two things; Receives (subcriber) the position of the sphere from 
		   Gazebo using the topic: '/gazebo/model_states'. 
		   Then it  publishes that position in the topic: 'marker_position' ; that topic updates the position 
		   of a  Marker (sphere) in rviz (just rviz NOT gazebo). A Marker was used because 
		   it was easier to work in RViz. The mark represents a goal in RViz

		   Executable name in the setup file: reader_mark_node
'''


import rclpy                 # The ROS2 Python library
from rclpy.node import Node  # Import the Node module and Enables the use of rclpy's Node class
from std_msgs.msg import String  
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker
import argparse


class MyNode(Node):

	def __init__(self):

		super().__init__('node_sphere_position_mark')
		

		self.marker_0_publisher    = self.create_publisher(Marker ,'marker_position_0', 10 )
		self.marker_1_publisher    = self.create_publisher(Marker ,'marker_position_1', 10 )
		self.marker_2_publisher    = self.create_publisher(Marker ,'marker_position_2', 10 )
		self.states_subscription = self.create_subscription(ModelStates, '/gazebo/model_states', self.state_lister_callback, 10)


	def state_lister_callback(self, msg):	

		try:
			# get the corret index for the sphere 
			sphere_index = msg.name.index('my_sphere_0') # have to do this cause if i include the robot arm the index may change

			# sphere position from Gazebo
			self.pos_0_x = msg.pose[sphere_index].position.x 
			self.pos_0_y = msg.pose[sphere_index].position.y 
			self.pos_0_z = msg.pose[sphere_index].position.z 

			sphere_index = msg.name.index('my_sphere_1') # have to do this cause if i include the robot arm the index may change

			# sphere position from Gazebo
			self.pos_1_x = msg.pose[sphere_index].position.x 
			self.pos_1_y = msg.pose[sphere_index].position.y 
			self.pos_1_z = msg.pose[sphere_index].position.z 
   
			sphere_index = msg.name.index('my_sphere_2') # have to do this cause if i include the robot arm the index may change

			# sphere position from Gazebo
			self.pos_2_x = msg.pose[sphere_index].position.x 
			self.pos_2_y = msg.pose[sphere_index].position.y 
			self.pos_2_z = msg.pose[sphere_index].position.z 

			#self.get_logger().info('Goal Position: X:%s Y:%s Z:%s ' % (round(self.pos_x,2), round(self.pos_y,2), round(self.pos_z,2) ))

		except:
			self.get_logger().info('could not get the sphere position yet, trying again...')

		
		else:
			# Publish msj with the mark's position in topic --> marker_position --> will need this later in the rviz

			marker_0 = Marker()
			marker_0.header.frame_id = '/world'
			marker_0.id = 0
			marker_0.type = marker_0.SPHERE
			marker_0.action = marker_0.ADD

			marker_0.pose.position.x = self.pos_0_x
			marker_0.pose.position.y = self.pos_0_y
			marker_0.pose.position.z = self.pos_0_z

			marker_0.pose.orientation.x = 0.0
			marker_0.pose.orientation.y = 0.0
			marker_0.pose.orientation.z = 0.0
			marker_0.pose.orientation.w = 1.0

			marker_0.scale.x = 0.15
			marker_0.scale.y = 0.15
			marker_0.scale.z = 0.15

			marker_0.color.a = 1.0
			marker_0.color.r = 0.1
			marker_0.color.g = 1.0
			marker_0.color.b = 0.0
   
			marker_1 = Marker()
			marker_1.header.frame_id = '/world'
			marker_1.id = 0
			marker_1.type = marker_1.SPHERE
			marker_1.action = marker_1.ADD

			marker_1.pose.position.x = self.pos_1_x
			marker_1.pose.position.y = self.pos_1_y
			marker_1.pose.position.z = self.pos_1_z

			marker_1.pose.orientation.x = 0.0
			marker_1.pose.orientation.y = 0.0
			marker_1.pose.orientation.z = 0.0
			marker_1.pose.orientation.w = 1.0

			marker_1.scale.x = 0.15
			marker_1.scale.y = 0.15
			marker_1.scale.z = 0.15

			marker_1.color.a = 1.0
			marker_1.color.r = 0.1
			marker_1.color.g = 1.0
			marker_1.color.b = 0.0

			marker_2 = Marker()
			marker_2.header.frame_id = '/world'
			marker_2.id = 0
			marker_2.type = marker_2.SPHERE
			marker_2.action = marker_2.ADD

			marker_2.pose.position.x = self.pos_2_x
			marker_2.pose.position.y = self.pos_2_y
			marker_2.pose.position.z = self.pos_2_z

			marker_2.pose.orientation.x = 0.0
			marker_2.pose.orientation.y = 0.0
			marker_2.pose.orientation.z = 0.0
			marker_2.pose.orientation.w = 1.0

			marker_2.scale.x = 0.15
			marker_2.scale.y = 0.15
			marker_2.scale.z = 0.15

			marker_2.color.a = 1.0
			marker_2.color.r = 0.1
			marker_2.color.g = 1.0
			marker_2.color.b = 0.0
			
			self.marker_0_publisher.publish(marker_0)
			self.marker_1_publisher.publish(marker_1)
			self.marker_2_publisher.publish(marker_2)



def main(args=None):

	rclpy.init(args=args)  # To initialize ROS communications

	node = MyNode()
	
	rclpy.spin(node)  # spin will be able to call any callback function that you’ve defined
					  # Also, waiting for kill the node with shutdown or Control C
	
	node.destroy_node()

	rclpy.shutdown()  # Shutdown what you started with rclpy.init(args=args)


if __name__ == '__main__':

	main()