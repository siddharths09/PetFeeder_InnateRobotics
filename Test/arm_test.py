#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import sys
import tty
import termios
import threading
import select

# Key mappings
INCREMENT_KEYS = ['1', '2', '3', '4', '5', '6']
DECREMENT_KEYS = ['q', 'w', 'e', 'r', 't', 'y']
JOINT_STEP = 0.065  # radians per key press

class ArmTeleopNode(Node):
    def __init__(self):
        super().__init__('arm_teleop')
        
        # Subscribe to arm state
        self.state_subscriber = self.create_subscription(
            JointState,
            '/mars/arm/state',
            self.state_callback,
            10
        )
        
        # Publisher to arm commands
        self.command_publisher = self.create_publisher(
            Float64MultiArray,
            '/mars/arm/commands',
            10
        )
        
        self.joint_positions = None
        self.got_joint_states = False
        
        self.running = True
        threading.Thread(target=self.keyboard_loop, daemon=True).start()
        
        self.get_logger().info("Arm Teleop Node started.")
        self.get_logger().info("Increment joints: 1-6 | Decrement joints: q,w,e,r,t,y | Ctrl+C to exit")

    def state_callback(self, msg: JointState):
        """Store the latest joint positions."""
        if msg.position:
            self.joint_positions = list(msg.position)
            self.got_joint_states = True
        else:
            self.get_logger().warn("Received JointState message with no position data.")

    def keyboard_loop(self):
        """Main keyboard input loop."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            print("\n" + "="*60)
            print("ARM TELEOP CONTROLLER")
            print("="*60)
            print("Increment joints: 1 2 3 4 5 6")
            print("Decrement joints: q w e r t y")
            print("Ctrl+C to exit")
            print("="*60 + "\n")
            
            while self.running:
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x03':  # Ctrl+C
                            self.running = False
                            break
                        self.handle_key(key)
                except KeyboardInterrupt:
                    self.running = False
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print("\nTeleop controller stopped.")

    def handle_key(self, key):
        """Process keyboard input and publish commands."""
        if not self.got_joint_states:
            print("Waiting for joint states from /mars/arm/state...")
            return
        
        if self.joint_positions is None:
            return
        
        new_positions = self.joint_positions.copy()
        joint_idx = -1
        direction = ""
        
        if key in INCREMENT_KEYS:
            joint_idx = INCREMENT_KEYS.index(key)
            new_positions[joint_idx] += JOINT_STEP
            direction = "+"
        elif key in DECREMENT_KEYS:
            joint_idx = DECREMENT_KEYS.index(key)
            new_positions[joint_idx] -= JOINT_STEP
            direction = "-"
        else:
            return  # Ignore other keys
        
        # Publish the new command
        command_msg = Float64MultiArray()
        command_msg.data = new_positions
        self.command_publisher.publish(command_msg)
        
        # Update stored positions
        self.joint_positions = new_positions
        
        # Print feedback
        print(f"Joint {joint_idx} {direction}{JOINT_STEP:.3f} rad → {new_positions[joint_idx]:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()