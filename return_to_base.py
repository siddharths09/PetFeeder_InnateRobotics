#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Twist
import tf2_ros
import transforms3d.euler as euler

class ReturnToBase(Node):
    def __init__(self):
        super().__init__('return_to_base')
        
        # Declare and get parameters
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('distance_buffer', 0.0)  # Extra distance to reverse
        
        self.linear_speed = self.get_parameter('linear_speed').value
        self.distance_buffer = self.get_parameter('distance_buffer').value
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.complete_pub = self.create_publisher(Bool, 'return_complete', 10)
        
        # Subscribe to travel distance from move_to_object
        self.distance_sub = self.create_subscription(
            Float64,
            'travel_distance',
            self.distance_callback,
            10
        )
        
        # Subscribe to arm grasp completion
        self.arm_complete_sub = self.create_subscription(
            Bool,
            'arm_grasp_complete',
            self.arm_complete_callback,
            10
        )
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State
        self.return_distance = None
        self.returning = False
        self.completed = False
        self.start_x = None
        self.start_y = None
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Return to Base Node initialized')
        self.get_logger().info('Waiting for travel_distance and arm_grasp_complete...')
        self.get_logger().info('=' * 50)
    
    def distance_callback(self, msg):
        """Receive travel distance from move_to_object"""
        self.return_distance = msg.data + self.distance_buffer
        self.get_logger().info(f'Received travel distance: {msg.data:.3f}m (will reverse {self.return_distance:.3f}m)')
    
    def arm_complete_callback(self, msg):
        """Callback when arm grasp is complete - start return to base"""
        if msg.data and not self.returning and not self.completed:
            if self.return_distance is None:
                self.get_logger().warn('No travel distance received! Cannot return.')
                return
            
            self.get_logger().info('=' * 50)
            self.get_logger().info(f'Arm grasp complete! Reversing {self.return_distance:.3f}m...')
            self.get_logger().info('=' * 50)
            
            # Record starting position
            pose = self.get_current_pose()
            if pose:
                self.start_x, self.start_y, _ = pose
                self.get_logger().info(f'Start position: ({self.start_x:.3f}, {self.start_y:.3f})')
            
            self.returning = True
            self.control_timer = self.create_timer(0.1, self.control_loop)
    
    def get_current_pose(self):
        """Get current robot pose from TF"""
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            
            roll, pitch, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])
            
            return x, y, yaw
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
    
    def control_loop(self):
        """Drive backwards for the received distance"""
        if not self.returning:
            return
        
        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().warn('[CONTROL] Lost robot pose')
            return
        
        x, y, theta = pose
        
        # Calculate distance traveled from start
        if self.start_x is None or self.start_y is None:
            self.start_x, self.start_y = x, y
            return
        
        dx = x - self.start_x
        dy = y - self.start_y
        distance_traveled = math.sqrt(dx**2 + dy**2)
        
        # Debug logging
        self.get_logger().info(f'[DEBUG] Traveled: {distance_traveled:.3f}m / {self.return_distance:.3f}m')
        
        # Check if we've traveled far enough
        if distance_traveled >= self.return_distance:
            self.finish_return()
            return
        
        # Drive backwards
        twist = Twist()
        twist.linear.x = -self.linear_speed
        twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(twist)
    
    def finish_return(self):
        """Complete the return to base sequence"""
        # Stop the robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        # Cancel control timer
        if hasattr(self, 'control_timer') and self.control_timer:
            self.control_timer.cancel()
        
        # Publish completion
        complete_msg = Bool()
        complete_msg.data = True
        self.complete_pub.publish(complete_msg)
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Successfully returned to base!')
        self.get_logger().info('=' * 50)
        
        self.returning = False
        self.completed = True

def main(args=None):
    rclpy.init(args=args)
    node = ReturnToBase()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()