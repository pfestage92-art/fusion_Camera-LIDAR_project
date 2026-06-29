#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np

class Lidar2DImage(Node):
    def __init__(self):
        super().__init__('lidar_2d_image')
        
        # Paramètres de filtrage (identiques à lidar_2d_projection)
        self.X_MIN = 0.0
        self.X_MAX = 0.5
        self.Y_MIN = -1.0
        self.Y_MAX = 1.0
        self.Z_MIN = 0.2
        self.Z_MAX = 5.0
        
        # Taille de l'image
        self.image_width = 640
        self.image_height = 480
        
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.sub = self.create_subscription(PointCloud2, '/unilidar/cloud', self.callback, 10)
        self.pub = self.create_publisher(Image, '/lidar_2d_image', 10)
        
        self.get_logger().info("LIDAR 2D IMAGE - Filtre + Projection sur image")
    
    def callback(self, msg):
        self.frame_count += 1
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        
        if len(points) == 0:
            return
        
        # Créer une image noire
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        points_projetes = 0
        for x, y, z, intensity in points:
            if (self.X_MIN < x < self.X_MAX and 
                self.Y_MIN < y < self.Y_MAX and 
                self.Z_MIN < z < self.Z_MAX):
                # Projection: (x, y, z) → (u, v) sur l'image
                u = int((y - self.Y_MIN) / (self.Y_MAX - self.Y_MIN) * self.image_width)
                v = int((x - self.X_MIN) / (self.X_MAX - self.X_MIN) * self.image_height)
                if 0 <= u < self.image_width and 0 <= v < self.image_height:
                    cv2.circle(img, (u, v), 2, (0, 255, 0), -1)
                    points_projetes += 1
        
        cv2.putText(img, f"Points: {points_projetes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        
        if self.frame_count % 50 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {len(points)} -> {points_projetes} points projetes")

def main(args=None):
    rclpy.init(args=args)
    node = Lidar2DImage()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
