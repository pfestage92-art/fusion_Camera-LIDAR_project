#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class SimpleLidarProjection(Node):
    def __init__(self):
        super().__init__('simple_lidar_projection')
        
        # Paramètres
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        self.declare_parameter('zoom', 1.0)
        self.declare_parameter('offset_u', 0)
        self.declare_parameter('offset_v', 0)
        self.declare_parameter('swap_xy', False)   # échange X et Y
        self.declare_parameter('invert_v', False)  # inversion verticale
        self.declare_parameter('point_size', 2)
        
        self.w = self.get_parameter('image_width').value
        self.h = self.get_parameter('image_height').value
        self.zoom = self.get_parameter('zoom').value
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value
        self.swap_xy = self.get_parameter('swap_xy').value
        self.invert_v = self.get_parameter('invert_v').value
        self.point_size = self.get_parameter('point_size').value
        
        self.bridge = CvBridge()
        self.image = None
        
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/unilidar/cloud', self.lidar_cb, 10)
        self.pub = self.create_publisher(Image, '/lidar/simple_projection', 10)
        
        self.get_logger().info("Simple Lidar Projection started")
        self.get_logger().info(f"Image size: {self.w}x{self.h}")
        self.get_logger().info(f"Zoom: {self.zoom}, Offsets: ({self.offset_u},{self.offset_v})")
        self.get_logger().info(f"swap_xy={self.swap_xy}, invert_v={self.invert_v}")
    
    def image_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image = cv2.resize(self.image, (self.w, self.h))
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")
    
    def lidar_cb(self, msg):
        if self.image is None:
            return
        
        points = list(pc2.read_points(msg, field_names=('x','y','z','intensity'), skip_nans=True))
        if len(points) == 0:
            return
        
        # Paramètres fixes (focale = largeur de l'image, approximation)
        fx = self.w
        fy = self.h
        cx = self.w // 2
        cy = self.h // 2
        
        vis = self.image.copy()
        proj_count = 0
        
        for (x, y, z, intensity) in points:
            # Échange éventuel des axes
            if self.swap_xy:
                u_coord = y
                v_coord = x
            else:
                u_coord = x
                v_coord = y
            
            # Projection (on utilise z comme distance)
            if z <= 0.0:
                continue
            u = int(fx * (u_coord / z) * self.zoom + cx + self.offset_u)
            v = int(fy * (v_coord / z) * self.zoom + cy + self.offset_v)
            
            if self.invert_v:
                v = self.h - v
            
            if 0 <= u < self.w and 0 <= v < self.h:
                # Couleur basée sur l'intensité
                r = min(255, int(intensity * 2))
                g = min(255, int(intensity * 1.2))
                b = min(255, int(intensity * 0.5))
                cv2.circle(vis, (u, v), self.point_size, (b, g, r), -1)
                proj_count += 1
        
        # Affichage du nombre de points projetés
        cv2.putText(vis, f"Points: {proj_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        self.pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {proj_count} points projetés")
        
        self.frame_count = getattr(self, 'frame_count', 0) + 1

def main(args=None):
    rclpy.init(args=args)
    node = SimpleLidarProjection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
