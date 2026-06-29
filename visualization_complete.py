#!/usr/bin/env python3
"""
VISUALISATION COMPLÈTE - CAMÉRA + LIDAR + FUSION
Comme dans l'image de référence
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class VisualizationComplete(Node):
    def __init__(self):
        super().__init__('visualization_complete')
        
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
        self.pub_viz = self.create_publisher(Image, '/visualization/complete', 10)
        
        self.get_logger().info("Visualisation Complète - Caméra + LiDAR + Fusion")
    
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
    
    def get_depth_color(self, distance):
        """Couleur pour estimation de profondeur (rouge = proche, bleu = loin)"""
        if distance < 2.0:
            return (0, 0, 255)    # Rouge (proche)
        elif distance < 5.0:
            return (0, 165, 255)  # Orange
        elif distance < 10.0:
            return (0, 255, 255)  # Jaune
        else:
            return (255, 0, 0)    # Bleu (loin)
    
    def lidar_callback(self, msg):
        if self.image_couleur is None:
            return
        
        self.frame_count += 1
        
        # FPS
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = current_time
        
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        
        # =============================================
        # IMAGE 1: CAMÉRA SEULE (coin haut-gauche)
        # =============================================
        img_camera = self.image_couleur.copy()
        cv2.putText(img_camera, "CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # =============================================
        # IMAGE 2: LIDAR PROJETÉ (coin haut-droit)
        # =============================================
        img_lidar = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        for p in points:
            x, y, z, intensity = p
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            if pixel is not None:
                u, v = pixel
                distance = math.sqrt(x*x + y*y + z*z)
                color = self.get_depth_color(distance)
                cv2.circle(img_lidar, (u, v), 2, color, -1)
        
        cv2.putText(img_lidar, "LIDAR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img_lidar, f"Points: {len(points)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # =============================================
        # IMAGE 3: FUSION + ESTIMATION PROFONDEUR (coin bas)
        # =============================================
        img_fusion = self.image_couleur.copy()
        
        for p in points:
            x, y, z, intensity = p
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            if pixel is not None:
                u, v = pixel
                distance = math.sqrt(x*x + y*y + z*z)
                color = self.get_depth_color(distance)
                
                # Si c'est une ligne, souligner en vert
                if self.mask is not None and self.mask[v, u] > 0:
                    cv2.circle(img_fusion, (u, v), 4, (0, 255, 0), -1)  # Vert
                    cv2.circle(img_fusion, (u, v), 2, color, -1)
                else:
                    cv2.circle(img_fusion, (u, v), 2, color, -1)
        
        cv2.putText(img_fusion, "FUSION + DEPTH ESTIMATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Légende profondeur
        cv2.putText(img_fusion, "Proche (<2m): Rouge", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img_fusion, "Loin (>10m): Bleu", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(img_fusion, "Lignes: Contour Vert", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # =============================================
        # ASSEMBLAGE DES 3 IMAGES
        # =============================================
        # Créer une grande image (2x largeur, 1.5x hauteur)
        h, w = self.image_height, self.image_width
        canvas = np.zeros((int(h * 1.5), w * 2, 3), dtype=np.uint8)
        
        # Placer les images
        canvas[0:h, 0:w] = img_camera
        canvas[0:h, w:w*2] = img_lidar
        canvas[h:h+int(h*0.5), 0:w*2] = cv2.resize(img_fusion, (w*2, int(h*0.5)))
        
        # Ajouter les FPS
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (10, canvas.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Publier
        img_msg = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
        self.pub_viz.publish(img_msg)
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {len(points)} points | FPS: {self.fps:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationComplete()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
