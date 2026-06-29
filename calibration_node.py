#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray
import numpy as np

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        
        # Paramètres intrinsèques
        self.fx = 2000.0
        self.fy = 2000.0
        self.cx = 1024.0
        self.cy = 768.0
        self.image_width = 2048
        self.image_height = 1536
        
        # Offsets
        self.offset_u = 50
        self.offset_v = -400
        
        # Matrice extrinsèque
        self.T = [0.0, -1.0, 0.0, 0.0,
                  1.0, 0.0, 0.0, 0.18,
                  0.0, 0.0, 1.0, 0.70,
                  0.0, 0.0, 0.0, 1.0]
        
        # Publishers
        self.camera_info_pub = self.create_publisher(CameraInfo, '/calibration/camera_info', 10)
        self.extrinsic_pub = self.create_publisher(Float64MultiArray, '/calibration/extrinsic', 10)
        self.offset_pub = self.create_publisher(Float64MultiArray, '/calibration/offsets', 10)
        
        self.timer = self.create_timer(0.1, self.publish_calibration)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("📷 NŒUD DE CALIBRATION DÉMARRÉ")
        self.get_logger().info(f"   fx={self.fx}, fy={self.fy}")
        self.get_logger().info(f"   cx={self.cx}, cy={self.cy}")
        self.get_logger().info(f"   offsets: u={self.offset_u}, v={self.offset_v}")
        self.get_logger().info("=" * 50)
    
    def publish_calibration(self):
        # 1. CameraInfo
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.width = self.image_width
        msg.height = self.image_height
        msg.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        self.camera_info_pub.publish(msg)
        
        # 2. Matrice extrinsèque
        ext_msg = Float64MultiArray()
        ext_msg.data = self.T
        self.extrinsic_pub.publish(ext_msg)
        
        # 3. Offsets
        off_msg = Float64MultiArray()
        off_msg.data = [float(self.offset_u), float(self.offset_v)]
        self.offset_pub.publish(off_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
