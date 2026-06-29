#!/usr/bin/env python3
"""
NŒUD LIDAR FULL OVERLAY
Affiche TOUS les points LiDAR projetés sur l'image
- Points sur les lignes : ROUGE
- Points sur les autres objets : COULEUR (selon distance ou intensité)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarFullOverlay(Node):
    def __init__(self):
        super().__init__('lidar_full_overlay')
        
        # Paramètres
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fov', 60.0)
        
        # Calibration
        self.declare_parameter('calib_x', 0.215)
        self.declare_parameter('calib_y', -0.06)
        self.declare_parameter('calib_z', 0.17)
        self.declare_parameter('calib_angle', 10.0)
        
        # Décalage
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
        self.mask = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        
        # Subscribers
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_mask = self.create_subscription(Image, '/detected/mask', self.mask_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/lidar_2d', self.lidar_callback, 10)
        
        # Publisher
        self.pub_overlay = self.create_publisher(Image, '/lidar_full_overlay', 10)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("LIDAR FULL OVERLAY")
        self.get_logger().info("Points sur lignes: ROUGE")
        self.get_logger().info("Points sur objets: COULEUR (selon distance)")
        self.get_logger().info("=" * 50)
    
    def image_callback(self, msg):
        try:
            self.image_couleur = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(self.image_couleur, (self.image_width, self.image_height))
        except:
            pass
    
    def mask_callback(self, msg):
        try:
            mask_cv = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            kernel = np.ones((3,3), np.uint8)
            mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)
            self.mask = cv2.resize(mask_cv, (self.image_width, self.image_height))
        except:
            pass
    
    def transform_lidar_to_camera(self, x, y, z):
        x_t = x - self.calib_x
        y_t = y - self.calib_y
        z_t = z - self.calib_z
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
        u = u + self.offset_u
        v = v + self.offset_v
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return (u, v)
        return None
    
    def get_color_by_distance(self, x, y, z):
        """Couleur basée sur la distance (proche = rouge, loin = bleu)"""
        distance = math.sqrt(x*x + y*y + z*z)
        
        if distance < 2.0:
            return (0, 0, 255)    # Rouge (très proche)
        elif distance < 4.0:
            return (0, 255, 255)  # Jaune (moyen)
        elif distance < 6.0:
            return (0, 255, 0)    # Vert
        else:
            return (255, 0, 0)    # Bleu (loin)
    
    def get_color_by_intensity(self, intensity):
        """Couleur basée sur l'intensité du retour LiDAR"""
        if intensity > 150:
            return (0, 0, 255)    # Rouge (forte intensité)
        elif intensity > 80:
            return (0, 255, 255)  # Jaune
        else:
            return (255, 255, 0)  # Cyan (faible intensité)
    
    def lidar_callback(self, msg):
        if self.image_couleur is None or self.mask is None:
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
        
        points_projetes = 0
        points_sur_lignes = 0
        
        for p in points:
            x, y, z, intensity = p
            
            # Transformation
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            
            # Projection
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            
            if pixel is not None:
                u, v = pixel
                points_projetes += 1
                
                # Vérifier si le point est sur une ligne
                if self.mask[v, u] > 0:
                    points_sur_lignes += 1
                    # POINTS SUR LES LIGNES : ROUGE
                    cv2.circle(vis_image, (u, v), 2, (0, 0, 255), -1)
                else:
                    # POINTS SUR AUTRES OBJETS : couleur selon distance
                    color = self.get_color_by_distance(x, y, z)
                    cv2.circle(vis_image, (u, v), 1, color, -1)
        
        # Dessiner les contours des lignes (vert)
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
        # Informations
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Points LiDAR: {points_projetes}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Points sur lignes: {points_sur_lignes}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Légende
        cv2.putText(vis_image, "Rouge: Lignes", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis_image, "Jaune: 2-4m", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, "Vert: 4-6m", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_image, "Bleu: >6m", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Publier
        img_msg = self.bridge.cv2_to_imgmsg(vis_image, 'bgr8')
        self.pub_overlay.publish(img_msg)
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(points)} LiDAR -> "
                f"{points_projetes} projetés, {points_sur_lignes} sur lignes | FPS: {self.fps:.1f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = LidarFullOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
