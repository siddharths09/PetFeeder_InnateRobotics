#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
import tf2_ros
import numpy as np
import transforms3d.euler as euler
import matplotlib.pyplot as plt
from std_msgs.msg import Float64


class MoveToObjectNode(Node):
    def __init__(self):
        super().__init__('move_to_object_node')
        
        # Declare parameters
        self.declare_parameter('distance_topic', '/object/base_position')
        self.declare_parameter('stop_distance', 0.05)
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_correction', 0.0)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('startup_delay', 2.0)
        self.declare_parameter('bezier_offset', 0.2)
        self.declare_parameter('num_waypoints', 10)
        self.declare_parameter('waypoint_tolerance', 0.05)
        self.declare_parameter('show_plot', True)
        
        # PID parameters
        self.declare_parameter('kp_angular', 1.0)
        self.declare_parameter('ki_angular', 0.0)
        self.declare_parameter('kd_angular', 0.3)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('angular_deadband', 0.02)  # Ignore small errors
        
        self.distance_topic = self.get_parameter('distance_topic').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_correction = self.get_parameter('angular_correction').value
        self.auto_start = self.get_parameter('auto_start').value
        self.startup_delay = self.get_parameter('startup_delay').value
        self.bezier_offset = self.get_parameter('bezier_offset').value
        self.num_waypoints = self.get_parameter('num_waypoints').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.show_plot = self.get_parameter('show_plot').value
        
        # PID gains
        self.kp_angular = self.get_parameter('kp_angular').value
        self.ki_angular = self.get_parameter('ki_angular').value
        self.kd_angular = self.get_parameter('kd_angular').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.angular_deadband = self.get_parameter('angular_deadband').value
        
        # PID state
        self.prev_angle_error = 0.0
        self.integral_error = 0.0
        self.last_control_time = None
        
        # Low-pass filter for angular velocity
        self.filtered_angular_vel = 0.0
        self.angular_filter_alpha = 0.3  # Lower = smoother, higher = more responsive
        
        # TF2 setup (for robot pose only)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.complete_pub = self.create_publisher(Bool, 'move_to_object_complete', 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.distance_topic,
            self.pose_callback,
            10
        )
        self.start_sub = self.create_subscription(
            Bool, 'move_to_object_start', self.start_callback, 10)
        
        self.distance_pub = self.create_publisher(Float64, 'travel_distance', 10)
        
        # State
        self.target_distance = None
        self.target_acquired = False
        self.moving = False
        self.control_timer = None
        self.messages_received = 0
        
        # Trajectory state
        self.waypoints = []
        self.current_waypoint_idx = 0
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Move To Object Node (Bézier Trajectory)')
        self.get_logger().info(f'Subscribing to: "{self.distance_topic}"')
        self.get_logger().info(f'Will stop {self.stop_distance}m from object')
        self.get_logger().info(f'PID: Kp={self.kp_angular}, Ki={self.ki_angular}, Kd={self.kd_angular}')
        self.get_logger().info(f'Max angular vel: {self.max_angular_vel} rad/s')
        self.get_logger().info('=' * 50)
        
        # Debug timer
        self.debug_timer = self.create_timer(3.0, self.debug_status)
        
        if self.auto_start:
            self.get_logger().info(f'Auto-start in {self.startup_delay} seconds...')
            self.startup_timer = self.create_timer(self.startup_delay, self.auto_start_movement)

        self.completed = False
    
    def bezier_curve(self, p0, p1, p2, p3, t):
        """Cubic Bézier interpolation between 4 control points."""
        return (1 - t)**3 * p0 + 3*(1 - t)**2*t*p1 + 3*(1 - t)*t**2*p2 + t**3*p3
    
    def generate_bezier_waypoints(self, x1, y1, theta1, x2, y2, theta2):
        """Generate (x, y, theta) waypoints along a smooth Bézier path."""
        d1 = np.array([np.cos(theta1), np.sin(theta1)])
        d2 = np.array([-np.cos(theta2), -np.sin(theta2)])
        
        c1 = np.array([x1, y1]) + self.bezier_offset * d1
        c2 = np.array([x2, y2]) + self.bezier_offset * d2
        
        t_vals = np.linspace(0, 1, self.num_waypoints)
        pts = [self.bezier_curve(np.array([x1, y1]), c1, c2, np.array([x2, y2]), t) for t in t_vals]
        
        thetas = []
        for i in range(len(pts) - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            thetas.append(np.arctan2(dy, dx))
        thetas.append(thetas[-1])
        
        return [(pts[i][0], pts[i][1], thetas[i]) for i in range(len(pts))]
    
    def plot_trajectory(self, target_x, target_y):
        """Visualize the planned trajectory."""
        x_vals = [p[0] for p in self.waypoints]
        y_vals = [p[1] for p in self.waypoints]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, '-o', label='Trajectory', markersize=4)
        
        plt.scatter(x_vals[0], y_vals[0], color='green', s=100, zorder=5, label='Start')
        plt.scatter(x_vals[-1], y_vals[-1], color='blue', s=100, zorder=5, label='Stop Position')
        plt.scatter(target_x, target_y, color='red', s=150, marker='x', zorder=5, label='Object')
        
        for x, y, theta in self.waypoints:
            plt.arrow(x, y, 0.03*np.cos(theta), 0.03*np.sin(theta),
                      head_width=0.01, head_length=0.005, fc='blue', ec='blue')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Bézier Trajectory to Object (stopping {self.stop_distance}m short)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
    
    def get_current_pose(self):
        """Get current robot pose from TF."""
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
            self.get_logger().warn(f'[TF] Could not get current pose: {e}')
            return None
    
    def plan_trajectory(self):
        """Plan Bézier trajectory from current pose to target distance."""
        current_pose = self.get_current_pose()
        if current_pose is None:
            self.get_logger().warn('Cannot plan trajectory: no current pose')
            return False
        
        x1, y1, theta1 = current_pose
        travel_distance = self.target_distance - self.stop_distance
        
        self.get_logger().info(f'Current pose: ({x1:.3f}, {y1:.3f}, {math.degrees(theta1):.1f}°)')
        self.get_logger().info(f'Object distance: {self.target_distance:.3f}m')
        self.get_logger().info(f'Distance to travel: {travel_distance:.3f}m')
        
        if travel_distance <= 0:
            self.get_logger().info('Already within stop distance of target!')
            return False
        
        x2 = x1 + travel_distance * math.cos(theta1)
        y2 = y1 + travel_distance * math.sin(theta1)
        theta2 = theta1
        
        target_x = x1 + self.target_distance * math.cos(theta1)
        target_y = y1 + self.target_distance * math.sin(theta1)
        
        self.waypoints = self.generate_bezier_waypoints(x1, y1, theta1, x2, y2, theta2)
        self.current_waypoint_idx = 0
        
        # Reset PID state
        self.prev_angle_error = 0.0
        self.integral_error = 0.0
        self.filtered_angular_vel = 0.0
        self.last_control_time = self.get_clock().now()
        
        self.get_logger().info(f'Planned {len(self.waypoints)} waypoints')
        self.get_logger().info(f'End position: ({x2:.3f}, {y2:.3f})')
        
        if self.show_plot:
            self.plot_trajectory(target_x, target_y)
        
        return True
    
    def debug_status(self):
        """Periodic debug output."""
        if not self.target_acquired:
            self.get_logger().warn(f'[DEBUG] Waiting for object detection...')
            self.get_logger().warn(f'[DEBUG] Topic: "{self.distance_topic}"')
            self.get_logger().warn(f'[DEBUG] Messages received: {self.messages_received}')
            self.get_logger().warn(f'[DEBUG] Make sure white object is visible to camera')
        elif self.moving:
            current_pose = self.get_current_pose()
            if current_pose:
                x, y, theta = current_pose
                self.get_logger().info(f'[DEBUG] Current: ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f}°)')
            self.get_logger().info(f'[DEBUG] Waypoint: {self.current_waypoint_idx + 1}/{len(self.waypoints)}')
    
    def pose_callback(self, msg):
        """Extract distance from pose z value (only once)."""
        if self.target_acquired:
            return
        
        self.messages_received += 1
        distance_to_object = msg.pose.position.z
        
        if distance_to_object <= 0:
            self.get_logger().warn(f'[DEBUG] Invalid distance: {distance_to_object:.3f}m')
            return
        
        self.target_distance = distance_to_object
        self.target_acquired = True
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Target acquired!')
        self.get_logger().info(f'Distance to object (z): {self.target_distance:.3f}m')
        self.get_logger().info('=' * 50)
        
        self.destroy_subscription(self.pose_sub)
    
    def auto_start_movement(self):
        self.startup_timer.cancel()
        self.start_moving()
    
    def start_callback(self, msg):
        if msg.data and not self.moving:
            self.start_moving()
    
    def start_moving(self):
        if self.moving or self.completed:
            return
        
        if not self.target_acquired:
            self.get_logger().warn('=' * 50)
            self.get_logger().warn('Cannot start: No target acquired!')
            self.get_logger().warn('Possible issues:')
            self.get_logger().warn('  1. White object not visible to camera')
            self.get_logger().warn('  2. Computer vision node not running')
            self.get_logger().warn('  3. Lighting conditions')
            self.get_logger().warn('Will retry in 2 seconds...')
            self.get_logger().warn('=' * 50)
            self.retry_timer = self.create_timer(2.0, self.retry_start)
            return
        
        if hasattr(self, 'retry_timer') and self.retry_timer:
            self.retry_timer.cancel()
        
        if not self.plan_trajectory():
            self.get_logger().warn('Failed to plan trajectory, retrying...')
            self.retry_timer = self.create_timer(2.0, self.retry_start)
            return
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Starting trajectory execution')
        self.get_logger().info('=' * 50)
        
        self.moving = True
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz for smoother control

    def retry_start(self):
        if hasattr(self, 'retry_timer') and self.retry_timer:
            self.retry_timer.cancel()
        if not self.moving and not self.completed:
            self.start_moving()
    
    def control_loop(self):
        """Follow waypoints using PID control with smoothing."""
        if not self.moving:
            return
        
        current_pose = self.get_current_pose()
        if current_pose is None:
            self.get_logger().warn('[CONTROL] Lost robot pose, stopping...')
            self.stop_robot()
            return
        
        x, y, theta = current_pose
        
        # Check if trajectory complete
        if self.current_waypoint_idx >= len(self.waypoints):
            self.finish_movement()
            return
        
        wx, wy, wtheta = self.waypoints[self.current_waypoint_idx]
        
        # Distance to current waypoint
        dx = wx - x
        dy = wy - y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Advance to next waypoint if close enough
        if dist < self.waypoint_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.waypoints)}')
            
            # Reset integral when changing waypoints
            self.integral_error = 0.0
            
            if self.current_waypoint_idx >= len(self.waypoints):
                self.finish_movement()
                return
            
            wx, wy, wtheta = self.waypoints[self.current_waypoint_idx]
            dx = wx - x
            dy = wy - y
            dist = math.sqrt(dx**2 + dy**2)
        
        # Calculate time delta
        current_time = self.get_clock().now()
        if self.last_control_time is None:
            dt = 0.05
        else:
            dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time
        
        # Clamp dt to reasonable range
        dt = max(0.01, min(dt, 0.2))
        
        # Calculate angle error
        angle_to_waypoint = math.atan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_waypoint - theta)
        
        # Apply deadband to reduce jitter
        if abs(angle_error) < self.angular_deadband:
            angle_error = 0.0
        
        # PID control for angular velocity
        # Proportional
        p_term = self.kp_angular * angle_error
        
        # Integral (with anti-windup)
        self.integral_error += angle_error * dt
        max_integral = 0.5  # Limit integral windup
        self.integral_error = max(-max_integral, min(max_integral, self.integral_error))
        i_term = self.ki_angular * self.integral_error
        
        # Derivative (on error, with filtering)
        derivative = (angle_error - self.prev_angle_error) / dt
        d_term = self.kd_angular * derivative
        self.prev_angle_error = angle_error
        
        # Raw angular velocity
        raw_angular_vel = p_term + i_term + d_term + self.angular_correction
        
        # Low-pass filter for smoothness
        self.filtered_angular_vel = (self.angular_filter_alpha * raw_angular_vel + 
                                      (1 - self.angular_filter_alpha) * self.filtered_angular_vel)
        
        # Clamp angular velocity
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, self.filtered_angular_vel))
        
        # Calculate linear velocity
        # Slow down when turning or approaching waypoint
        turn_factor = max(0.4, 1.0 - abs(angle_error) / (math.pi / 2))
        approach_factor = min(1.0, dist / 0.2)  # Slow down within 0.2m of waypoint
        linear_vel = self.linear_speed * turn_factor * approach_factor
        
        # Minimum linear velocity to keep moving
        linear_vel = max(0.05, linear_vel)
        
        # Create and publish twist message
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(twist_msg)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def stop_robot(self):
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
    
    def finish_movement(self):
        self.stop_robot()
        self.completed = True
        
        if self.control_timer:
            self.control_timer.cancel()
        
        if self.debug_timer:
            self.debug_timer.cancel()
        
        if hasattr(self, 'retry_timer') and self.retry_timer:
            self.retry_timer.cancel()
        
        # Publish travel distance
        distance_msg = Float64()
        distance_msg.data = self.target_distance - self.stop_distance
        self.distance_pub.publish(distance_msg)
        self.get_logger().info(f'Published travel distance: {distance_msg.data:.3f}m')
        
        complete_msg = Bool()
        complete_msg.data = True
        self.complete_pub.publish(complete_msg)
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Trajectory complete!')
        self.get_logger().info('=' * 50)
        
        if self.show_plot:
            plt.close('all')
        
        self.moving = False
        self.get_logger().info('Node idle. Ctrl+C to exit.')

def main(args=None):
    rclpy.init(args=args)
    node = MoveToObjectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist_msg = Twist()
        node.cmd_vel_pub.publish(twist_msg)
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()