import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.control()

    def control(self):
        LINEAR_SPEED = 0.5
        ANGULAR_SPEED = 0.5

        while rclpy.ok():
            try:
                key=input("Enter your command (w/a/s/d) or q to quit: ")
                twist_msg= Twist()

                if key.lower()=='w':
                    twist_msg.linear.x = LINEAR_SPEED
                elif key.lower()=='s':
                    twist_msg.linear.x = -LINEAR_SPEED
                elif key.lower()=='a':
                    twist_msg.angular.z = ANGULAR_SPEED
                elif key.lower()=='d':
                    twist_msg.angular.z = -ANGULAR_SPEED
                else:
                    self.get_logger.warn('INVALID KEY')
                    continue

                self.publisher_.publish(twist_msg)
                
            except EOFError:
                break

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
