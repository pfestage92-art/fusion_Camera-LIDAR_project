#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarRawProjection(Node):
    def __init__(self):
        super().__init__('lidar_raw_projection')
        
        # Paramètres image
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('point_size', 2)
        self.declare_parameter('decay_time', 5.0)
        
        # Paramètres de projection (zoom et offsets)
        self.declare_parameter('zoom', 1.0)               # facteur d'échelle
        self.declare_parameter('offset_u', 0)             # décalage horizontal (pixels)
        self.declare_parameter('offset_v', 0)             # décalage vertical (pixels)
        
        # Calibration extrinsèque (translation + rotation)
        self.declare_parameter('calib_x', 0.0)
        self.declare_parameter('calib_y', 0.0)
        self.declare_parameter('calib_z', 0.0)
        self.declare_parameter('calib_angle', 0.0)        # rotation autour de Z (degrés)
        
        # Permutation et inversion d'axes
        self.declare_parameter('swap_xy', False)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('invert_y', False)
        self.declare_parameter('invert_z', False)
        
        # Inversion verticale de l'image (haut ↔ bas)
        self.declare_parameter('invert_v', False)
        
        # FOV pour le calcul de focale
        self.declare_parameter('camera_fov', 60.0)
        
        # Récupération des paramètres
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.point_size = self.get_parameter('point_size').value
        self.decay_time = self.get_parameter('decay_time').value
        self.zoom = self.get_parameter('zoom').value
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value
        self.calib_x = self.get_parameter('calib_x').value
        self.calib_y = self.get_parameter('calib_y').value
        self.calib_z = self.get_parameter('calib_z').value
        self.calib_angle = math.radians(self.get_parameter('calib_angle').value)
        self.swap_xy = self.get_parameter('swap_xy').value
        self.invert_x = self.get_parameter('invert_x').value
        self.invert_y = self.get_parameter('invert_y').value
        self.invert_z = self.get_parameter('invert_z').value
        self.invert_v = self.get_parameter('invert_v').value
        
        # Calcul des paramètres intrinsèques (focale et centre)
        fov_rad = math.radians(self.get_parameter('camera_fov').value)
        self.fx = self.image_width / (2 * math.tan(fov_rad / 2))
        self.fy = self.image_height / (2 * math.tan(fov_rad / 2))
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        self.bridge = CvBridge()
        self.image_couleur = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        self.points_accumules = []          # pour la persistance
        
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/unilidar/cloud', self.lidar_callback, 10)
        self.pub = self.create_publisher(Image, '/lidar/raw_projection', 10)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("LIDAR RAW PROJECTION (version corrigée)")
        self.get_logger().info(f"Image: {self.image_width}x{self.image_height}")
        self.get_logger().info(f"Zoom: {self.zoom}, offsets: u={self.offset_u}, v={self.offset_v}")
        self.get_logger().info("=" * 60)
    
    def image_callback(self, msg):
        try:
            self.image_couleur = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(self.image_couleur, (self.image_width, self.image_height))
        except Exception as e:
            self.get_logger().error(f"Erreur image: {e}")
    
    def transform_point(self, x, y, z):
        """Applique rotation, permutation, translation pour aligner LiDAR -> caméra"""
        # Permutation des axes
        if self.swap_xy:
            x, y = y, x
        if self.invert_x:
            x = -x
        if self.invert_y:
            y = -y
        if self.invert_z:
            z = -z
        
        # Translation
        x_t = x - self.calib_x
        y_t = y - self.calib_y
        z_t = z - self.calib_z
        
        # Rotation autour de Z
        cos_a = math.cos(self.calib_angle)
        sin_a = math.sin(self.calib_angle)
        x_cam = x_t * cos_a - y_t * sin_a
        y_cam = x_t * sin_a + y_t * cos_a
        z_cam = z_t
        
        return x_cam, y_cam, z_cam
    
    def project_to_image(self, x, y, z):
        """Projection perspective 3D → 2D"""
        if z <= 0.01:          # point derrière ou trop proche
            return None
        # Projection standard puis application du zoom et des offsets
        u = int(self.fx * (x / z) * self.zoom + self.cx + self.offset_u)
        v = int(self.fy * (y / z) * self.zoom + self.cy + self.offset_v)
        return (u, v)
    
    def get_color(self, intensity):
        """Couleur basée sur l'intensité du retour LiDAR"""
        # Dégradé : faible intensité → bleu, forte → rouge
        r = min(255, int(intensity * 1.5))
        g = min(255, int(intensity * 0.8))
        b = min(255, int(255 - intensity))
        return (b, g, r)   # BGR pour OpenCV
    
    def lidar_callback(self, msg):
        if self.image_couleur is None:
            return
        
        self.frame_count += 1
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Calcul du FPS
        dt = (current_time - self.last_time.nanoseconds / 1e9) if hasattr(self.last_time, 'nanoseconds') else 0.1
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = self.get_clock().now()
        
        # Lire tous les points LiDAR
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if len(points) == 0:
            return
        
        # Afficher un échantillon pour déboguer
        if self.frame_count % 100 == 0 and len(points) > 0:
            x, y, z, intensity = points[0]
            self.get_logger().info(f"Point brut: x={x:.3f}, y={y:.3f}, z={z:.3f}, intensité={intensity}")
        
        nouveaux_points = []
        points_projetes = 0
        
        for (x, y, z, intensity) in points:
            # Transformation LiDAR -> caméra
            xc, yc, zc = self.transform_point(x, y, z)
            # Projection
            pix = self.project_to_image(xc, yc, zc)
            if pix is not None:
                u, v = pix
                points_projetes += 1
                # Optionnel : log du premier point projeté
                if points_projetes == 1 and self.frame_count % 100 == 0:
                    self.get_logger().info(f"Point projeté: u={u}, v={v}")
                color = self.get_color(intensity)
                nouveaux_points.append((u, v, color, current_time))
        
        # Mettre à jour la persistance
        self.points_accumules.extend(nouveaux_points)
        self.points_accumules = [(u, v, c, t) for (u, v, c, t) in self.points_accumules
                                  if current_time - t < self.decay_time]
        
        # Inversion verticale de l'image (si demandée)
        vis_image = self.image_couleur.copy()
        for (u, v, color, _) in self.points_accumules:
            # Appliquer inversion verticale à l'affichage
            v_display = v
            if self.invert_v:
                v_display = self.image_height - v
            if 0 <= u < self.image_width and 0 <= v_display < self.image_height:
                cv2.circle(vis_image, (u, v_display), self.point_size, color, -1)
        
        # Affichage des statistiques
        cv2.putText(vis_image, f"Points projetés: {len(self.points_accumules)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(vis_image, f"Zoom: {self.zoom:.3f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        self.pub.publish(self.bridge.cv2_to_imgmsg(vis_image, 'bgr8'))
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(nouveaux_points)} points projetés, "
                f"total visibles={len(self.points_accumules)}, FPS={self.fps:.1f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = LidarRawProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
