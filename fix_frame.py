#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class FixFrame(Node):
    def __init__(self):
        super().__init__('fix_frame')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Image, '/camera/image_fixed', 10)
        self.get_logger().info("FixFrame node started - correcting frame_id to camera_frame")
    
    def callback(self, msg):
        msg.header.frame_id = "camera_frame"   # ← correction ici
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FixFrame()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
