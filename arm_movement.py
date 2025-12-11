#!/usr/bin/env python3
"""
Arm Motion Server - Top-down vision grasping strategy
1. Open gripper
2. Move arm to overhead view position (camera looking down)
3. Detect object with arm camera
4. Position gripper above object
5. Lower and close gripper
6. Return to rest position
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Bool, Float64MultiArray
from maurice_msgs.srv import GotoJS
import math
import time
from enum import Enum

class GraspState(Enum):
    IDLE = 0
    OPEN_GRIPPER = 1
    MOVE_TO_OVERHEAD = 2
    WAIT_FOR_OVERHEAD_MOTION = 3
    SETTLE_AT_OVERHEAD = 4
    WAIT_FOR_ARM_CAMERA_DETECTION = 5
    POSITION_ABOVE_OBJECT = 6
    WAIT_FOR_POSITION_MOTION = 7
    LOWER_TO_OBJECT = 8
    WAIT_FOR_LOWER_MOTION = 9
    CLOSE_GRIPPER = 10
    RETURN_TO_REST = 11
    WAIT_FOR_REST_MOTION = 12

class ArmMotionServer(Node):
    def __init__(self):
        super().__init__('arm_motion_server')
        
        # ════════════════════════════════════════════════════════════════
        # CONFIGURATION - EDIT THESE VALUES
        # ════════════════════════════════════════════════════════════════
        
        # MODE SELECTION
        USE_JOINT_CONTROL = True  # True = direct joints, False = IK/Cartesian
        
        # OVERHEAD POSITION (Direct Joint Control)
        # Adjust these to get camera pointing down correctly
        OVERHEAD_JOINT1 = 0.386   # Base rotation
        OVERHEAD_JOINT2 = -0.257  # Shoulder
        OVERHEAD_JOINT3 = -0.5    # Elbow  
        OVERHEAD_JOINT4 = 1.8     # Wrist pitch (adjust this for camera angle!)
        OVERHEAD_JOINT5 = 0.4     # Wrist roll
        
        # OVERHEAD POSITION (IK/Cartesian - used if USE_JOINT_CONTROL=False)
        OVERHEAD_X = 0.20           # Forward from base
        OVERHEAD_Y = 0.00           # Left/right from base
        OVERHEAD_HEIGHT = 0.25      # Height above base
        OVERHEAD_WRIST_PITCH = 1.57 # Wrist angle in radians
        
        # GRASP SETTINGS
        GRASP_HEIGHT_OFFSET = 0.02  # How far above object to position (meters)
        GRASP_X_OFFSET = -0.01       # X offset from detected object position (meters)
        GRASP_Y_OFFSET = 0.0        # Y offset from detected object position (meters)
        GRASP_DEPTH = 0.07          # How far below object surface to lower for grasp (meters)
        
        MOTION_DURATION = 3.0       # How long each motion takes (seconds)
        
        # GRIPPER SETTINGS
        GRIPPER_OPEN_VALUE = 1.0    # Gripper open position
        GRIPPER_CLOSE_VALUE = 0.0   # Gripper close position  
        GRIPPER_DURATION = 2.0      # Time to wait for gripper (seconds)
        
        # DETECTION SETTINGS
        DETECTION_TIMEOUT = 5.0     # Max time to wait for camera detection (seconds)
        DETECTION_SETTLE_TIME = 2.0 # Time to wait after reaching overhead before starting detection (seconds)
        
        # ════════════════════════════════════════════════════════════════
        
        # Parameters (using config values as defaults)
        self.declare_parameter('overhead_joint1', OVERHEAD_JOINT1)
        self.declare_parameter('overhead_joint2', OVERHEAD_JOINT2)
        self.declare_parameter('overhead_joint3', OVERHEAD_JOINT3)
        self.declare_parameter('overhead_joint4', OVERHEAD_JOINT4)
        self.declare_parameter('overhead_joint5', OVERHEAD_JOINT5)
        self.declare_parameter('use_joint_control', USE_JOINT_CONTROL)
        
        self.declare_parameter('overhead_height', OVERHEAD_HEIGHT)
        self.declare_parameter('overhead_x', OVERHEAD_X)
        self.declare_parameter('overhead_y', OVERHEAD_Y)
        self.declare_parameter('overhead_wrist_pitch', OVERHEAD_WRIST_PITCH)
        
        self.declare_parameter('grasp_height_offset', GRASP_HEIGHT_OFFSET)
        self.declare_parameter('grasp_x_offset', GRASP_X_OFFSET)
        self.declare_parameter('grasp_y_offset', GRASP_Y_OFFSET)
        self.declare_parameter('grasp_depth', GRASP_DEPTH)
        self.declare_parameter('motion_duration', MOTION_DURATION)
        self.declare_parameter('gripper_open_value', GRIPPER_OPEN_VALUE)
        self.declare_parameter('gripper_close_value', GRIPPER_CLOSE_VALUE)
        self.declare_parameter('gripper_duration', GRIPPER_DURATION)
        self.declare_parameter('detection_timeout', DETECTION_TIMEOUT)
        self.declare_parameter('detection_settle_time', DETECTION_SETTLE_TIME)
        
        # Get parameters
        self.overhead_joint1 = self.get_parameter('overhead_joint1').value
        self.overhead_joint2 = self.get_parameter('overhead_joint2').value
        self.overhead_joint3 = self.get_parameter('overhead_joint3').value
        self.overhead_joint4 = self.get_parameter('overhead_joint4').value
        self.overhead_joint5 = self.get_parameter('overhead_joint5').value
        self.use_joint_control = self.get_parameter('use_joint_control').value
        
        self.overhead_height = self.get_parameter('overhead_height').value
        self.overhead_x = self.get_parameter('overhead_x').value
        self.overhead_y = self.get_parameter('overhead_y').value
        self.overhead_wrist_pitch = self.get_parameter('overhead_wrist_pitch').value
        self.grasp_height_offset = self.get_parameter('grasp_height_offset').value
        self.grasp_x_offset = self.get_parameter('grasp_x_offset').value
        self.grasp_y_offset = self.get_parameter('grasp_y_offset').value
        self.grasp_depth = self.get_parameter('grasp_depth').value
        self.motion_duration = self.get_parameter('motion_duration').value
        self.gripper_open_value = self.get_parameter('gripper_open_value').value
        self.gripper_close_value = self.get_parameter('gripper_close_value').value
        self.gripper_duration = self.get_parameter('gripper_duration').value
        self.detection_timeout = self.get_parameter('detection_timeout').value
        self.detection_settle_time = self.get_parameter('detection_settle_time').value
        
        # State machine
        self.state = GraspState.IDLE
        self.motion_start_time = None
        self.detection_wait_start = None
        self.grasp_attempted = False
        
        # Store initial arm position for return
        self.initial_joint_positions = None
        
        # Detection tracking
        self.arm_camera_detection = None  # From arm camera
        self.current_pose = None
        self.ik_solution = None
        self.latest_joint_state = None
        self.object_position_in_base = None  # Calculated object position
        
        # Subscribers
        self.create_subscription(
            PointStamped,
            '/object/arm_position',  # Now using ARM camera
            self.arm_camera_callback,
            10
        )
        
        self.create_subscription(
            PoseStamped,
            '/fk_pose',
            self.fk_pose_callback,
            10
        )
        
        # Subscribe to mobile base completion signal
        self.create_subscription(
            Bool,
            '/move_to_object_complete',
            self.mobile_base_complete_callback,
            10
        )
        
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        ik_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.create_subscription(
            JointState,
            '/ik_solution',
            self.ik_solution_callback,
            ik_qos
        )
        
        self.create_subscription(
            JointState,
            '/mars/arm/state',
            self.joint_state_callback,
            10
        )
        
        # Publishers
        self.ik_request_pub = self.create_publisher(Twist, '/ik_delta', 10)
        self.status_pub = self.create_publisher(String, '/arm_motion/status', 10)
        self.arm_command_pub = self.create_publisher(Float64MultiArray, '/mars/arm/commands', 10)
        
        # Completion signal for next node
        self.complete_pub = self.create_publisher(Bool, '/arm_grasp_complete', 10)
        
        # Service client
        self.goto_js_client = self.create_client(GotoJS, '/mars/arm/goto_js')
        
        # Timers
        self.create_timer(0.05, self.state_machine_update)  # 20Hz
        # Don't automatically trigger - wait for mobile base to complete
        # self.create_timer(2.0, self.trigger_grasp)  # Removed auto-trigger
        
        # Startup
        time.sleep(0.5)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Arm Motion Server - Top-Down Vision Strategy")
        self.get_logger().info("=" * 60)
        
        if self.use_joint_control:
            self.get_logger().info("Mode: JOINT CONTROL (direct joint positions)")
            self.get_logger().info(f"Overhead joints: J1={self.overhead_joint1:.2f}, J2={self.overhead_joint2:.2f}, "
                                  f"J3={self.overhead_joint3:.2f}, J4={self.overhead_joint4:.2f}, J5={self.overhead_joint5:.2f}")
        else:
            self.get_logger().info("Mode: IK CONTROL (Cartesian positioning)")
            self.get_logger().info(f"Overhead position: ({self.overhead_x:.2f}, {self.overhead_y:.2f}, {self.overhead_height:.2f})")
            self.get_logger().info(f"Wrist pitch angle: {self.overhead_wrist_pitch:.2f} rad ({math.degrees(self.overhead_wrist_pitch):.1f}°)")
        
        self.get_logger().info(f"Grasp offsets: Z={self.grasp_height_offset}m, X={self.grasp_x_offset}m, Y={self.grasp_y_offset}m")
        self.get_logger().info(f"Grasp depth: {self.grasp_depth}m (how far below object to lower)")
        
        self.get_logger().info(f"Gripper: open={self.gripper_open_value}, close={self.gripper_close_value}")
        self.get_logger().info(f"Gripper duration: {self.gripper_duration}s")
        
        if self.goto_js_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("✓ goto_js service available")
        
        wait_time = 0
        while self.current_pose is None and wait_time < 3.0:
            time.sleep(0.5)
            wait_time += 0.5
        
        if self.current_pose:
            p = self.current_pose.pose.position
            self.get_logger().info(f"✓ FK available: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        
        self.get_logger().info("Waiting for mobile base to complete...")
        self.get_logger().info("Listening on: /move_to_object_complete")
        self.get_logger().info("=" * 60)

    def arm_camera_callback(self, msg: PointStamped):
        """Receive object detection from arm camera"""
        self.arm_camera_detection = msg
        
        # Only log when actively waiting
        if self.state == GraspState.WAIT_FOR_ARM_CAMERA_DETECTION:
            self.get_logger().info(
                f"Arm camera detected object at: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})",
                throttle_duration_sec=1.0
            )

    def fk_pose_callback(self, msg: PoseStamped):
        self.current_pose = msg

    def ik_solution_callback(self, msg: JointState):
        self.ik_solution = msg

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg
        
        # Capture initial position on first callback
        if self.initial_joint_positions is None and len(msg.position) >= 6:
            self.initial_joint_positions = list(msg.position)
            self.get_logger().info(f"Captured initial position: {[f'{j:.2f}' for j in self.initial_joint_positions]}")

    def mobile_base_complete_callback(self, msg: Bool):
        """Triggered when mobile base finishes moving to object"""
        if msg.data and not self.grasp_attempted and self.state == GraspState.IDLE:
            self.get_logger().info("=" * 60)
            self.get_logger().info("✓ Mobile base complete! Starting grasp sequence...")
            self.get_logger().info("=" * 60)
            self.trigger_grasp()

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(status)

    def trigger_grasp(self):
        """Start the grasp sequence"""
        if self.state == GraspState.IDLE and not self.grasp_attempted:
            self.publish_status("Starting grasp sequence...")
            self.grasp_attempted = True
            self.open_gripper()

    def open_gripper(self):
        """Open gripper"""
        self.get_logger().info("STEP 1: Opening gripper...")
        self.get_logger().info(f"  Setting gripper to {self.gripper_open_value}")
        
        if self.latest_joint_state and len(self.latest_joint_state.position) >= 5:
            joints = list(self.latest_joint_state.position[:5])
            joints.append(self.gripper_open_value)
            
            cmd = Float64MultiArray()
            cmd.data = joints
            
            # Publish multiple times to ensure it's received
            for _ in range(5):
                self.arm_command_pub.publish(cmd)
                time.sleep(0.05)
            
            self.get_logger().info(f"  Sent command: {[f'{j:.2f}' for j in joints]}")
        else:
            self.get_logger().warn("No joint state available, cannot open gripper!")
        
        self.motion_start_time = time.time()
        self.state = GraspState.OPEN_GRIPPER

    def move_to_overhead_position(self):
        """Move to overhead viewing position"""
        self.get_logger().info("STEP 2: Moving to overhead position...")
        
        if self.use_joint_control:
            # Direct joint control - move to specific joint angles
            self.get_logger().info(f"  Using JOINT CONTROL")
            self.get_logger().info(f"  Joints: [{self.overhead_joint1:.2f}, {self.overhead_joint2:.2f}, "
                                  f"{self.overhead_joint3:.2f}, {self.overhead_joint4:.2f}, {self.overhead_joint5:.2f}]")
            
            # Build joint command with gripper open
            joints = [
                self.overhead_joint1,
                self.overhead_joint2,
                self.overhead_joint3,
                self.overhead_joint4,
                self.overhead_joint5,
                self.gripper_open_value
            ]
            
            # Send via goto_js service
            if self.goto_js_client.service_is_ready():
                request = GotoJS.Request()
                request.data.data = joints
                request.time = int(self.motion_duration)
                
                self.goto_js_client.call_async(request)
                self.get_logger().info(f"  Sent joint command: {[f'{j:.2f}' for j in joints]}")
                
                self.motion_start_time = time.time()
                self.state = GraspState.WAIT_FOR_OVERHEAD_MOTION
            else:
                self.get_logger().error("goto_js service not ready!")
                self.state = GraspState.IDLE
                self.grasp_attempted = False
        else:
            # IK control - request IK for Cartesian position
            self.get_logger().info(f"  Using IK CONTROL")
            self.get_logger().info(f"  Target: ({self.overhead_x:.3f}, {self.overhead_y:.3f}, {self.overhead_height:.3f})")
            self.get_logger().info(f"  Wrist pitch: {self.overhead_wrist_pitch:.2f} rad ({math.degrees(self.overhead_wrist_pitch):.1f}°)")
            
            twist = Twist()
            twist.linear.x = self.overhead_x
            twist.linear.y = self.overhead_y
            twist.linear.z = self.overhead_height
            twist.angular.x = 0.0
            twist.angular.y = self.overhead_wrist_pitch
            twist.angular.z = 0.0
            
            self.ik_solution = None
            for _ in range(3):
                self.ik_request_pub.publish(twist)
            
            self.motion_start_time = time.time()
            self.state = GraspState.MOVE_TO_OVERHEAD

    def calculate_object_position_in_base(self):
        """
        Calculate object position in base_link frame from arm camera detection.
        Arm camera is pointing down, so transform accordingly.
        """
        if self.arm_camera_detection is None or self.current_pose is None:
            return None
        
        # Object position in arm camera frame
        cam_x = self.arm_camera_detection.point.x
        cam_y = self.arm_camera_detection.point.y
        cam_z = self.arm_camera_detection.point.z
        
        # Current arm end-effector position in base frame
        ee_x = self.current_pose.pose.position.x
        ee_y = self.current_pose.pose.position.y
        ee_z = self.current_pose.pose.position.z
        
        # Transform: Camera pointing down, so:
        # cam_z (depth/down) gives us vertical offset
        # cam_x, cam_y give us horizontal offset
        
        # Object position in base frame
        obj_x = ee_x + cam_x  # Horizontal offset
        obj_y = ee_y - cam_y  # Horizontal offset (inverted)
        obj_z = ee_z - cam_z  # Vertical - camera is above, object below
        
        self.get_logger().info(f"Calculated object in base: ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})")
        
        return (obj_x, obj_y, obj_z)

    def position_above_object(self):
        """Position gripper above detected object"""
        self.get_logger().info("STEP 5: Positioning above object...")
        
        obj_pos = self.calculate_object_position_in_base()
        if obj_pos is None:
            self.get_logger().error("Failed to calculate object position!")
            self.state = GraspState.IDLE
            self.grasp_attempted = False
            return
        
        obj_x, obj_y, obj_z = obj_pos
        
        # Apply X and Y offsets to the detected position
        obj_x += self.grasp_x_offset
        obj_y += self.grasp_y_offset
        
        self.object_position_in_base = (obj_x, obj_y, obj_z)
        
        # Position above object with Z offset
        target_x = obj_x
        target_y = obj_y
        target_z = obj_z + self.grasp_height_offset
        
        self.get_logger().info(f"  Detected: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        self.get_logger().info(f"  Offset: X={self.grasp_x_offset:.3f}, Y={self.grasp_y_offset:.3f}")
        self.get_logger().info(f"  Target: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
        
        # Request IK (same wrist orientation as overhead)
        twist = Twist()
        twist.linear.x = target_x
        twist.linear.y = target_y
        twist.linear.z = target_z
        twist.angular.x = 0.0
        twist.angular.y = self.overhead_wrist_pitch  # Keep same wrist angle
        twist.angular.z = 0.0
        
        self.ik_solution = None
        for _ in range(3):
            self.ik_request_pub.publish(twist)
        
        self.motion_start_time = time.time()
        self.state = GraspState.POSITION_ABOVE_OBJECT

    def lower_to_object(self):
        """Lower gripper onto object"""
        self.get_logger().info("STEP 6: Lowering to object...")
        
        if self.object_position_in_base is None:
            self.get_logger().error("No object position!")
            return
        
        obj_x, obj_y, obj_z = self.object_position_in_base
        
        # Lower to grasp depth below object surface
        target_z = obj_z - self.grasp_depth
        
        self.get_logger().info(f"  Object Z: {obj_z:.3f}")
        self.get_logger().info(f"  Grasp depth: {self.grasp_depth:.3f}")
        self.get_logger().info(f"  Target Z: {target_z:.3f}")
        
        twist = Twist()
        twist.linear.x = obj_x
        twist.linear.y = obj_y
        twist.linear.z = target_z
        twist.angular.x = 0.0
        twist.angular.y = self.overhead_wrist_pitch  # Keep same wrist angle
        twist.angular.z = 0.0
        
        self.ik_solution = None
        for _ in range(3):
            self.ik_request_pub.publish(twist)
        
        self.motion_start_time = time.time()
        self.state = GraspState.LOWER_TO_OBJECT

    def close_gripper(self):
        """Close gripper on object"""
        self.get_logger().info("STEP 7: Closing gripper...")
        self.get_logger().info(f"  Setting gripper to {self.gripper_close_value}")
        
        if self.latest_joint_state and len(self.latest_joint_state.position) >= 5:
            joints = list(self.latest_joint_state.position[:5])
            joints.append(self.gripper_close_value)
            
            cmd = Float64MultiArray()
            cmd.data = joints
            
            # Publish multiple times to ensure it's received
            for _ in range(5):
                self.arm_command_pub.publish(cmd)
                time.sleep(0.05)
            
            self.get_logger().info(f"  Sent command: {[f'{j:.2f}' for j in joints]}")
        else:
            self.get_logger().warn("No joint state available, cannot close gripper!")
        
        self.motion_start_time = time.time()
        self.state = GraspState.CLOSE_GRIPPER

    def return_to_rest(self):
        """Return arm to initial rest position"""
        self.get_logger().info("STEP 8: Returning to initial position...")
        
        if self.initial_joint_positions is None:
            self.get_logger().error("No initial position saved!")
            self.state = GraspState.IDLE
            return
        
        request = GotoJS.Request()
        request.data.data = self.initial_joint_positions
        request.time = int(self.motion_duration)
        
        self.goto_js_client.call_async(request)
        
        self.motion_start_time = time.time()
        self.state = GraspState.RETURN_TO_REST

    def execute_ik_motion(self, next_state):
        """Execute motion from current IK solution - gripper stays in current state"""
        if self.ik_solution is None:
            self.get_logger().warn("No IK solution available")
            return
        
        joints = list(self.ik_solution.position)
        
        # Add current gripper position (DON'T change it during arm motion)
        if len(joints) == 5:
            if self.latest_joint_state and len(self.latest_joint_state.position) >= 6:
                # Keep current gripper position
                current_gripper = self.latest_joint_state.position[5]
                joints.append(current_gripper)
            else:
                # Fallback - keep gripper open
                joints.append(self.gripper_open_value)
        
        request = GotoJS.Request()
        request.data.data = joints
        request.time = int(self.motion_duration)
        
        self.get_logger().info(f"Executing: {[f'{j:.2f}' for j in joints]}")
        self.goto_js_client.call_async(request)
        
        self.motion_start_time = time.time()
        self.state = next_state

    def state_machine_update(self):
        """Main state machine (20Hz)"""
        
        if self.state == GraspState.OPEN_GRIPPER:
            if time.time() - self.motion_start_time > 1.5:
                self.move_to_overhead_position()
        
        elif self.state == GraspState.MOVE_TO_OVERHEAD:
            if self.ik_solution is not None:
                self.execute_ik_motion(GraspState.WAIT_FOR_OVERHEAD_MOTION)
            elif time.time() - self.motion_start_time > 3.0:
                self.get_logger().error("Overhead IK timeout")
                self.state = GraspState.IDLE
                self.grasp_attempted = False
        
        elif self.state == GraspState.WAIT_FOR_OVERHEAD_MOTION:
            if time.time() - self.motion_start_time > self.motion_duration + 0.5:
                self.get_logger().info(f"STEP 3: Settling at overhead position ({self.detection_settle_time}s)...")
                self.motion_start_time = time.time()
                self.arm_camera_detection = None  # Reset detection
                self.state = GraspState.SETTLE_AT_OVERHEAD
        
        elif self.state == GraspState.SETTLE_AT_OVERHEAD:
            # Wait for settle time before starting detection
            if time.time() - self.motion_start_time > self.detection_settle_time:
                self.get_logger().info("STEP 4: Waiting for arm camera detection...")
                self.detection_wait_start = time.time()
                self.arm_camera_detection = None  # Reset again for fresh detection
                self.state = GraspState.WAIT_FOR_ARM_CAMERA_DETECTION
        
        elif self.state == GraspState.WAIT_FOR_ARM_CAMERA_DETECTION:
            # Check for detection
            if self.arm_camera_detection is not None:
                # Got detection!
                age = time.time() - self.arm_camera_detection.header.stamp.sec
                if age < 1.0:  # Fresh detection
                    self.position_above_object()
            
            # Timeout check
            if time.time() - self.detection_wait_start > self.detection_timeout:
                self.get_logger().error("No object detected by arm camera!")
                self.state = GraspState.IDLE
                self.grasp_attempted = False
        
        elif self.state == GraspState.POSITION_ABOVE_OBJECT:
            if self.ik_solution is not None:
                self.execute_ik_motion(GraspState.WAIT_FOR_POSITION_MOTION)
            elif time.time() - self.motion_start_time > 3.0:
                self.get_logger().error("Position IK timeout")
                self.state = GraspState.IDLE
                self.grasp_attempted = False
        
        elif self.state == GraspState.WAIT_FOR_POSITION_MOTION:
            if time.time() - self.motion_start_time > self.motion_duration + 0.5:
                self.lower_to_object()
        
        elif self.state == GraspState.LOWER_TO_OBJECT:
            if self.ik_solution is not None:
                self.execute_ik_motion(GraspState.WAIT_FOR_LOWER_MOTION)
            elif time.time() - self.motion_start_time > 3.0:
                self.get_logger().error("Lower IK timeout")
                self.state = GraspState.IDLE
                self.grasp_attempted = False
        
        elif self.state == GraspState.WAIT_FOR_LOWER_MOTION:
            if time.time() - self.motion_start_time > self.motion_duration + 0.5:
                self.close_gripper()
        
        elif self.state == GraspState.CLOSE_GRIPPER:
            if time.time() - self.motion_start_time > 2.0:
                self.return_to_rest()
        
        elif self.state == GraspState.RETURN_TO_REST:
            # Motion already started in return_to_rest()
            self.state = GraspState.WAIT_FOR_REST_MOTION
        
        elif self.state == GraspState.WAIT_FOR_REST_MOTION:
            if time.time() - self.motion_start_time > self.motion_duration + 0.5:
                self.publish_status("✓ GRASP COMPLETE!")
                
                # Signal completion to next node
                complete_msg = Bool()
                complete_msg.data = True
                self.complete_pub.publish(complete_msg)
                
                self.get_logger().info("=" * 60)
                self.get_logger().info("Published completion signal on /arm_grasp_complete")
                self.get_logger().info("Next node can now start!")
                self.get_logger().info("=" * 60)
                
                self.state = GraspState.IDLE


def main(args=None):
    rclpy.init(args=args)
    node = ArmMotionServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()