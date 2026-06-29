#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarCalibrationProjection(Node):
    def __init__(self):
        super().__init__('lidar_calibration_projection')
        
        # =============================================
        # PARAMÈTRES DE L'IMAGE
        # =============================================
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        
        # =============================================
        # PARAMÈTRES DE PROJECTION (zoom et offsets)
        # =============================================
        self.declare_parameter('zoom', 0.001)           # facteur d'échelle
        self.declare_parameter('offset_u', 1024)        # décalage horizontal (pixels)
        self.declare_parameter('offset_v', 768)         # décalage vertical (pixels)
        
        # =============================================
        # CALIBRATION EXTRINSÈQUE (LiDAR → Caméra)
        # =============================================
        self.declare_parameter('calib_x', 0.215)        # translation X (devant)
        self.declare_parameter('calib_y', -0.06)        # translation Y (gauche)
        self.declare_parameter('calib_z', 0.17)         # translation Z (haut)
        self.declare_parameter('calib_angle', 10.0)     # rotation autour de Z (degrés)
        
        # =============================================
        # PERMUTATION ET INVERSION D'AXES
        # =============================================
        self.declare_parameter('swap_xy', False)        # échange X et Y
        self.declare_parameter('invert_x', False)       # inverse X
        self.declare_parameter('invert_y', False)       # inverse Y
        self.declare_parameter('invert_z', False)       # inverse Z
        
        # =============================================
        # INVERSION VERTICALE DE L'IMAGE
        # =============================================
        self.declare_parameter('invert_v', False)       # haut ↔ bas
        
        # =============================================
        # FILTRAGE SPATIAL (pour ne garder que certains points)
        # =============================================
        self.declare_parameter('x_min', -100.0)
        self.declare_parameter('x_max', 100.0)
        self.declare_parameter('y_min', -100.0)
        self.declare_parameter('y_max', 100.0)
        self.declare_parameter('z_min', -100.0)
        self.declare_parameter('z_max', 100.0)
        
        # =============================================
        # VISUALISATION
        # =============================================
        self.declare_parameter('point_size', 3)
        self.declare_parameter('decay_time', 5.0)
        self.declare_parameter('camera_fov', 60.0)      # pour le calcul des focales
        
        # Récupération des paramètres
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
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
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.point_size = self.get_parameter('point_size').value
        self.decay_time = self.get_parameter('decay_time').value
        self.camera_fov = math.radians(self.get_parameter('camera_fov').value)
        
        # Calcul des focales (caméra pinhole)
        self.fx = self.image_width / (2 * math.tan(self.camera_fov / 2))
        self.fy = self.image_height / (2 * math.tan(self.camera_fov / 2))
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        self.bridge = CvBridge()
        self.image = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        self.points_accumules = []          # pour la persistance
        
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/unilidar/cloud', self.lidar_callback, 10)
        self.pub = self.create_publisher(Image, '/lidar/calibration_projection', 10)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("LIDAR CALIBRATION PROJECTION - TOUS PARAMÈTRES")
        self.get_logger().info(f"Image: {self.image_width}x{self.image_height}")
        self.get_logger().info(f"Zoom: {self.zoom}, offsets: u={self.offset_u}, v={self.offset_v}")
        self.get_logger().info(f"Calibration: X={self.calib_x}, Y={self.calib_y}, Z={self.calib_z}, angle={self.get_parameter('calib_angle').value}°")
        self.get_logger().info(f"swap_xy={self.swap_xy}, invert_v={self.invert_v}")
        self.get_logger().info("=" * 60)
    
    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image = cv2.resize(self.image, (self.image_width, self.image_height))
        except:
            pass
    
    def transform_point(self, x, y, z):
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
        if z <= 0.01:
            return None
        u = int(self.fx * (x / z) * self.zoom + self.cx + self.offset_u)
        v = int(self.fy * (y / z) * self.zoom + self.cy + self.offset_v)
        if self.invert_v:
            v = self.image_height - v
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return (u, v)
        return None
    
    def get_color(self, intensity):
        r = min(255, int(intensity * 1.5))
        g = min(255, int(intensity * 0.8))
        b = min(255, int(255 - intensity))
        return (b, g, r)
    
    def lidar_callback(self, msg):
        if self.image is None:
            return
        
        self.frame_count += 1
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # FPS
        dt = (current_time - self.last_time.nanoseconds / 1e9) if hasattr(self.last_time, 'nanoseconds') else 0.1
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = self.get_clock().now()
        
        # Lire les points LiDAR
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if len(points) == 0:
            return
        
        nouveaux_points = []
        points_projetes = 0
        
        for (x, y, z, intensity) in points:
            # Filtrage spatial
            if not (self.x_min < x < self.x_max and self.y_min < y < self.y_max and self.z_min < z < self.z_max):
                continue
            
            xc, yc, zc = self.transform_point(x, y, z)
            pixel = self.project_to_image(xc, yc, zc)
            if pixel is not None:
                u, v = pixel
                points_projetes += 1
                color = self.get_color(intensity)
                nouveaux_points.append((u, v, color, current_time))
        
        self.points_accumules.extend(nouveaux_points)
        self.points_accumules = [(u, v, c, t) for (u, v, c, t) in self.points_accumules 
                                  if current_time - t < self.decay_time]
        
        vis_image = self.image.copy()
        for (u, v, color, _) in self.points_accumules:
            cv2.circle(vis_image, (u, v), self.point_size, color, -1)
        
        # Affichage des statistiques
        cv2.putText(vis_image, f"Points: {len(self.points_accumules)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        self.pub.publish(self.bridge.cv2_to_imgmsg(vis_image, 'bgr8'))
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {points_projetes} points projetés, total visibles={len(self.points_accumules)}, FPS={self.fps:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarCalibrationProjection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
