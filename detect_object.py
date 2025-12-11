#!/usr/bin/env python3
"""
Object Detection Publisher - Detects objects and publishes their positions
Both base and arm cameras with compressed images
No motion control - separate servers handle that
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String, Bool
import cv2
import numpy as np
from cv_bridge import CvBridge

class ObjectDetectionPublisher(Node):
    def __init__(self):
        super().__init__('object_detection_publisher')
        
        self.bridge = CvBridge()
        
        # Detection parameters
        self.declare_parameter('white_threshold', 100)
        self.declare_parameter('min_contour_area', 50)
        self.declare_parameter('object_real_size', 0.03)  # Real object size (meters)
        
        # Compression parameters
        self.declare_parameter('jpeg_quality', 80)  # JPEG quality (0-100)
        
        # Camera intrinsic parameters
        self.declare_parameter('base_camera.fx', 800.0)
        self.declare_parameter('base_camera.fy', 800.0)
        self.declare_parameter('base_camera.cx', 640.0)
        self.declare_parameter('base_camera.cy', 400.0)
        self.declare_parameter('arm_camera.fx', 400.0)
        self.declare_parameter('arm_camera.fy', 400.0)
        self.declare_parameter('arm_camera.cx', 320.0)
        self.declare_parameter('arm_camera.cy', 240.0)
        
        # Get parameters
        self.white_threshold = self.get_parameter('white_threshold').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.object_real_size = self.get_parameter('object_real_size').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        
        self.base_cam = {
            'fx': self.get_parameter('base_camera.fx').value,
            'fy': self.get_parameter('base_camera.fy').value,
            'cx': self.get_parameter('base_camera.cx').value,
            'cy': self.get_parameter('base_camera.cy').value,
        }
        
        self.arm_cam = {
            'fx': self.get_parameter('arm_camera.fx').value,
            'fy': self.get_parameter('arm_camera.fy').value,
            'cx': self.get_parameter('arm_camera.cx').value,
            'cy': self.get_parameter('arm_camera.cy').value,
        }
        
        # State
        self.base_image = None
        self.arm_image = None
        
        # QoS for images
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.create_subscription(
            Image, '/mars/main_camera/image',
            self.base_image_callback, image_qos
        )
        self.create_subscription(
            Image, '/mars/arm/image_raw',
            self.arm_image_callback, image_qos
        )
        
        # Publishers
        # Object position from base camera (for mobile base server)
        self.base_object_pub = self.create_publisher(
            PoseStamped,
            '/object/base_position',
            10
        )
        
        # Object position from arm camera (for arm motion server)
        self.arm_object_pub = self.create_publisher(
            PointStamped,
            '/object/arm_position',
            10
        )
        
        # Detection status
        self.detection_status_pub = self.create_publisher(
            String,
            '/object/detection_status',
            10
        )
        
        # Object detected flag (simple bool)
        self.object_detected_pub = self.create_publisher(
            Bool,
            '/object/detected',
            10
        )
        
        # Debug images (RAW)
        self.base_debug_pub = self.create_publisher(
            Image, '/object_detection/base_debug', 10
        )
        self.arm_debug_pub = self.create_publisher(
            Image, '/object_detection/arm_debug', 10
        )
        
        # Debug images (COMPRESSED - for Foxglove over network)
        self.base_debug_compressed_pub = self.create_publisher(
            CompressedImage, '/object_detection/base_debug/compressed', 10
        )
        self.arm_debug_compressed_pub = self.create_publisher(
            CompressedImage, '/object_detection/arm_debug/compressed', 10
        )
        
        # Processing timer
        self.create_timer(0.1, self.process_frames)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Object Detection Publisher Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Detection Parameters:")
        self.get_logger().info(f"  - White Threshold: {self.white_threshold}")
        self.get_logger().info(f"  - Min Contour Area: {self.min_contour_area}")
        self.get_logger().info(f"  - Object Size: {self.object_real_size}m")
        self.get_logger().info(f"Image Compression:")
        self.get_logger().info(f"  - JPEG Quality: {self.jpeg_quality}")
        self.get_logger().info("Publishing Topics:")
        self.get_logger().info("  - /object/base_position")
        self.get_logger().info("  - /object/arm_position")
        self.get_logger().info("  - /object/detection_status")
        self.get_logger().info("  - /object/detected")
        self.get_logger().info("  - /object_detection/base_debug/compressed")
        self.get_logger().info("  - /object_detection/arm_debug/compressed")
        self.get_logger().info("=" * 60)

    def base_image_callback(self, msg: Image):
        try:
            self.base_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Base image conversion failed: {e}")

    def arm_image_callback(self, msg: Image):
        try:
            self.arm_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Arm image conversion failed: {e}")

    def detect_white_object(self, image, camera_params, camera_name="unknown"):
        """
        Detect white objects in image
        BASE camera: only lower half (ground objects)
        ARM camera: full image (close-up manipulation)
        Returns detection dict with distance estimation
        """
        if image is None:
            return None
        
        # Get image dimensions
        h_img, w_img = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # ═══════════════════════════════════════════════════════
        # FILTER: BASE camera = lower half only, ARM camera = full image
        # ═══════════════════════════════════════════════════════
        if camera_name == "BASE":
            # Only look at lower half for ground objects
            mask = np.zeros_like(binary)
            mask[h_img//2:, :] = 255  # Only bottom half
            binary = cv2.bitwise_and(binary, mask)
        # ARM camera uses full image (no mask)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest valid contour
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        if not valid_contours:
            return None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get properties
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(largest_contour)
        area = cv2.contourArea(largest_contour)
        
        # ═══════════════════════════════════════════════════════
        # DISTANCE ESTIMATION (Similar triangles method)
        # ═══════════════════════════════════════════════════════
        pixel_size = max(w, h)
        f = (camera_params['fx'] + camera_params['fy']) / 2.0
        distance = (self.object_real_size * f) / pixel_size if pixel_size > 0 else 0
        
        # Sanity check: clamp to reasonable range
        distance = np.clip(distance, 0.1, 5.0)  # Between 10cm and 5m
        
        # Convert to 3D point in camera frame
        x_norm = (cx - camera_params['cx']) / camera_params['fx']
        y_norm = (cy - camera_params['cy']) / camera_params['fy']
        point_3d = np.array([x_norm * distance, y_norm * distance, distance])
        
        return {
            'center': (cx, cy),
            'bbox': (x, y, w, h),
            'area': area,
            'contour': largest_contour,
            'distance': distance,
            'point_3d_camera': point_3d,
        }

    def process_frames(self):
        """Process both camera feeds and publish detections"""
        
        # ═══════════════════════════════════════════════════════
        # BASE CAMERA DETECTION (for mobile base positioning)
        # ═══════════════════════════════════════════════════════
        base_detection = self.detect_white_object(self.base_image, self.base_cam, "BASE")
        
        if base_detection is not None:
            # Publish object position from base camera
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_camera_optical_frame"
            
            point_3d = base_detection['point_3d_camera']
            pose_msg.pose.position.x = point_3d[0]
            pose_msg.pose.position.y = point_3d[1]
            pose_msg.pose.position.z = point_3d[2]
            
            # Default orientation (pointing forward)
            pose_msg.pose.orientation.w = 1.0
            
            self.base_object_pub.publish(pose_msg)
            
            # Publish detection flag
            detected_msg = Bool()
            detected_msg.data = True
            self.object_detected_pub.publish(detected_msg)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"BASE_DETECTED: distance={base_detection['distance']:.3f}m"
            self.detection_status_pub.publish(status_msg)
            
            self.get_logger().info(
                f"BASE: Object at ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}m)",
                throttle_duration_sec=1.0
            )
        else:
            # No detection
            detected_msg = Bool()
            detected_msg.data = False
            self.object_detected_pub.publish(detected_msg)
        
        # ═══════════════════════════════════════════════════════
        # ARM CAMERA DETECTION (for fine arm positioning)
        # ═══════════════════════════════════════════════════════
        arm_detection = self.detect_white_object(self.arm_image, self.arm_cam, "ARM")
        
        if arm_detection is not None:
            # Publish object position from arm camera
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "arm_camera_optical_frame"
            
            point_3d = arm_detection['point_3d_camera']
            point_msg.point.x = point_3d[0]
            point_msg.point.y = point_3d[1]
            point_msg.point.z = point_3d[2]
            
            self.arm_object_pub.publish(point_msg)
            
            # Publish status with pixel coordinates
            cx, cy = arm_detection['center']
            status_msg = String()
            status_msg.data = f"ARM_DETECTED: pixel=({cx},{cy}), distance={arm_detection['distance']:.3f}m"
            self.detection_status_pub.publish(status_msg)
            
            self.get_logger().info(
                f"ARM: Object at pixel ({cx}, {cy}), distance={arm_detection['distance']:.3f}m",
                throttle_duration_sec=1.0
            )
        
        # ═══════════════════════════════════════════════════════
        # PUBLISH DEBUG IMAGES (Both RAW and COMPRESSED)
        # ═══════════════════════════════════════════════════════
        self.publish_debug_image(self.base_image, base_detection, "BASE")
        self.publish_debug_image(self.arm_image, arm_detection, "ARM")

    def publish_debug_image(self, image, detection, camera_name):
        """Create and publish debug visualization (both raw and compressed)"""
        if image is None:
            return
        
        debug_img = image.copy()
        h_img, w_img = image.shape[:2]
        
        # Draw detection region boundary (only for BASE camera)
        if camera_name == "BASE":
            cv2.line(debug_img, (0, h_img//2), (w_img, h_img//2), 
                     (255, 0, 255), 2)  # Magenta line
            cv2.putText(debug_img, "DETECTION ZONE (GROUND - LOWER HALF)", (10, h_img//2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            # ARM camera - full image detection
            cv2.putText(debug_img, "DETECTION ZONE (FULL IMAGE)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if detection is not None:
            cx, cy = detection['center']
            x, y, w, h = detection['bbox']
            
            # Draw detection
            cv2.drawContours(debug_img, [detection['contour']], -1, (0, 255, 0), 3)
            cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), -1)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add distance text
            dist_text = f"Distance: {detection['distance']:.3f}m"
            cv2.putText(debug_img, dist_text, (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add crosshair at centroid
            cv2.line(debug_img, (cx-30, cy), (cx+30, cy), (0, 255, 255), 2)
            cv2.line(debug_img, (cx, cy-30), (cx, cy+30), (0, 255, 255), 2)
            
            # Add pixel coordinates
            coord_text = f"({cx}, {cy})"
            cv2.putText(debug_img, coord_text, (cx+15, cy-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            # No detection message - placed lower to avoid overlap with zone text
            no_detect_y = 60 if camera_name == "ARM" else 30
            cv2.putText(debug_img, f"{camera_name}: No object detected", (10, no_detect_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add camera name and parameters
        info_y = 90 if camera_name == "ARM" else 60
        cv2.putText(debug_img, f"Camera: {camera_name}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Threshold: {self.white_threshold}", (10, info_y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Min Area: {self.min_contour_area}", (10, info_y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection mode indicator
        mode_text = "Mode: Lower Half Only" if camera_name == "BASE" else "Mode: Full Image"
        cv2.putText(debug_img, mode_text, (10, info_y+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ═══════════════════════════════════════════════════════
        # PUBLISH RAW IMAGE (for local debugging)
        # ═══════════════════════════════════════════════════════
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            if camera_name == "BASE":
                self.base_debug_pub.publish(debug_msg)
            else:
                self.arm_debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Raw debug image publish failed: {e}")
        
        # ═══════════════════════════════════════════════════════
        # PUBLISH COMPRESSED IMAGE (for Foxglove over network)
        # ═══════════════════════════════════════════════════════
        try:
            # Encode as JPEG with specified quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, compressed_data = cv2.imencode('.jpg', debug_img, encode_param)
            
            if not success:
                self.get_logger().error("Failed to encode image to JPEG")
                return
            
            # Create compressed image message
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.header.frame_id = f"{camera_name.lower()}_camera"
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed_data.tobytes()
            
            # Publish to appropriate topic
            if camera_name == "BASE":
                self.base_debug_compressed_pub.publish(compressed_msg)
            else:
                self.arm_debug_compressed_pub.publish(compressed_msg)
                
        except Exception as e:
            self.get_logger().error(f"Compressed image publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nShutting down Object Detection Publisher...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()