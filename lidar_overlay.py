#!/usr/bin/env python3
"""
NŒUD LIDAR OVERLAY
Affiche TOUS les points LiDAR bruts projetés sur l'image vidéo
Sans filtrage, sans détection de lignes
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarOverlay(Node):
    def __init__(self):
        super().__init__('lidar_overlay')
        
        # Paramètres
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fov', 60.0)
        
        # Calibration (distances caméra ↔ LiDAR)
        self.declare_parameter('calib_x', 0.215)   # LiDAR devant caméra (21.5 cm)
        self.declare_parameter('calib_y', -0.06)   # LiDAR à gauche (6 cm)
        self.declare_parameter('calib_z', 0.17)    # LiDAR au-dessus (17 cm)
        self.declare_parameter('calib_angle', 10.0) # Rotation (10 degrés)
        
        # Décalage manuel pour alignement
        self.declare_parameter('offset_u', 50)
        self.declare_parameter('offset_v', 200)
        
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.camera_fov = math.radians(self.get_parameter('camera_fov').value)
        
        self.calib_x = self.get_parameter('calib_x').value
        self.calib_y = self.get_parameter('calib_y').value
        self.calib_z = self.get_parameter('calib_z').value
        self.calib_angle = math.radians(self.get_parameter('calib_angle').value)
        
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value
        
        # Matrice de projection
        self.fx = self.image_width / (2 * math.tan(self.camera_fov / 2))
        self.fy = self.image_height / (2 * math.tan(self.camera_fov / 2))
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        self.bridge = CvBridge()
        self.image_couleur = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        
        # Subscribers
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/lidar_2d', self.lidar_callback, 10)
        
        # Publisher
        self.pub_overlay = self.create_publisher(Image, '/lidar_overlay/image', 10)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("LIDAR OVERLAY - TOUS LES POINTS LIDAR")
        self.get_logger().info(f"Offset: u={self.offset_u}, v={self.offset_v}")
        self.get_logger().info("Publication sur: /lidar_overlay/image")
        self.get_logger().info("=" * 50)
    
    def image_callback(self, msg):
        try:
            self.image_couleur = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(self.image_couleur, (self.image_width, self.image_height))
        except:
            pass
    
    def transform_lidar_to_camera(self, x, y, z):
        # Translation
        x_t = x - self.calib_x
        y_t = y - self.calib_y
        z_t = z - self.calib_z
        
        # Rotation
        cos_a = math.cos(self.calib_angle)
        sin_a = math.sin(self.calib_angle)
        
        x_cam = x_t * cos_a - y_t * sin_a
        y_cam = x_t * sin_a + y_t * cos_a
        z_cam = z_t
        
        return x_cam, y_cam, z_cam
    
    def project_to_image(self, x, y, z):
        if x <= 0.01:
            return None
        u = int(self.fx * (y / x) + self.cx)
        v = int(self.fy * (z / x) + self.cy)
        
        # Appliquer décalage
        u = u + self.offset_u
        v = v + self.offset_v
        
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return (u, v)
        return None
    
    def lidar_callback(self, msg):
        if self.image_couleur is None:
            return
        
        self.frame_count += 1
        
        # Calcul FPS
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = current_time
        
        # Lire tous les points LiDAR
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        
        if len(points) == 0:
            return
        
        # Copier l'image
        vis_image = self.image_couleur.copy()
        
        # Projeter TOUS les points
        points_projetes = 0
        for p in points:
            x, y, z, intensity = p
            
            # Transformation LiDAR → Caméra
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            
            # Projection dans l'image
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            
            if pixel is not None:
                u, v = pixel
                points_projetes += 1
                
                # Couleur selon l'intensité (rouge = forte intensité, bleu = faible)
                if intensity > 100:
                    color = (0, 0, 255)  # Rouge
                elif intensity > 50:
                    color = (0, 255, 255)  # Jaune
                else:
                    color = (255, 255, 0)  # Cyan
                
                cv2.circle(vis_image, (u, v), 1, color, -1)
        
        # Ajouter les informations
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Points LiDAR: {points_projetes}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Publier l'image
        img_msg = self.bridge.cv2_to_imgmsg(vis_image, 'bgr8')
        self.pub_overlay.publish(img_msg)
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {len(points)} points -> {points_projetes} projetés | FPS: {self.fps:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
