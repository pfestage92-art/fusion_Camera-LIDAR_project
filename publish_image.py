#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        
        self.declare_parameter('image_path', '/home/amin/runs/finfin1.jpeg')
        self.declare_parameter('loop', True)
        self.declare_parameter('fps', 10)
        
        image_path = self.get_parameter('image_path').value
        self.loop = self.get_parameter('loop').value
        self.fps = self.get_parameter('fps').value
        
        self.img = cv2.imread(image_path)
        if self.img is None:
            self.get_logger().error(f"❌ Image non trouvée: {image_path}")
            return
        
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.publish_image)
        
        self.get_logger().info(f"✅ Image publisher démarré")
        self.get_logger().info(f"   📷 Image: {image_path}")
        self.get_logger().info(f"   📐 Taille: {self.img.shape[1]}x{self.img.shape[0]}")
    
    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.img, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
