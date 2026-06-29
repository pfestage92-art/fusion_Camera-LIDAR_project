#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import json
from scipy.spatial.transform import Rotation as R

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        
        # ============================================
        # PARAMÈTRES DE CALIBRATION
        # ============================================
        # Paramètres intrinsèques
        self.declare_parameter('intrinsic_fx', 2000.29)
        self.declare_parameter('intrinsic_fy', 2000.31)
        self.declare_parameter('intrinsic_cx', 1024.76)
        self.declare_parameter('intrinsic_cy', 768.41)
        self.declare_parameter('intrinsic_width', 2024)
        self.declare_parameter('intrinsic_height', 3032)
        
        # Fichier de calibration extrinsèque
        self.declare_parameter('extrinsic_file', '/home/amin/calib_result/calib.json')
        
        # ============================================
        # PARAMÈTRES EXTRINSÈQUES (Distance caméra-LiDAR)
        # ============================================
        self.declare_parameter('calib_x', 0.215)     # Translation X (mètres)
        self.declare_parameter('calib_y', -0.06)     # Translation Y (mètres)
        self.declare_parameter('calib_z', 0.17)      # Translation Z (mètres)
        self.declare_parameter('calib_angle', 10.0)  # Rotation (degrés)
        
        # ============================================
        # PARAMÈTRES DE FUSION
        # ============================================
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        self.declare_parameter('camera_fov', 60.0)
        self.declare_parameter('x_min', 0.0)
        self.declare_parameter('x_max', 10.0)
        self.declare_parameter('y_min', -10.0)
        self.declare_parameter('y_max', 10.0)
        self.declare_parameter('offset_u', 100)
        self.declare_parameter('offset_v', -200)
        
        # Chargement des paramètres intrinsèques
        self.fx_intrinsic = self.get_parameter('intrinsic_fx').value
        self.fy_intrinsic = self.get_parameter('intrinsic_fy').value
        self.cx_intrinsic = self.get_parameter('intrinsic_cx').value
        self.cy_intrinsic = self.get_parameter('intrinsic_cy').value
        self.img_width_intrinsic = self.get_parameter('intrinsic_width').value
        self.img_height_intrinsic = self.get_parameter('intrinsic_height').value
        
        # Chargement des paramètres extrinsèques (DISTANCE CAMÉRA-LiDAR)
        self.calib_x = self.get_parameter('calib_x').value   # Translation X
        self.calib_y = self.get_parameter('calib_y').value   # Translation Y
        self.calib_z = self.get_parameter('calib_z').value   # Translation Z
        self.calib_angle = math.radians(self.get_parameter('calib_angle').value)  # Rotation en radians
        
        # Chargement des paramètres de fusion
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.camera_fov = math.radians(self.get_parameter('camera_fov').value)
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value
        
        # ============================================
        # CONSTRUCTION DE LA MATRICE EXTRINSÈQUE
        # ============================================
        # Translation
        T_vec = np.array([self.calib_x, self.calib_y, self.calib_z])
        
        # Rotation (autour de l'axe Z pour calib_angle)
        cos_a = math.cos(self.calib_angle)
        sin_a = math.sin(self.calib_angle)
        R_mat = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Matrice de transformation 4x4
        self.T_lidar_camera = np.eye(4)
        self.T_lidar_camera[:3, :3] = R_mat
        self.T_lidar_camera[:3, 3] = T_vec
        
        # Matrices de projection
        self.fx = self.image_width / (2 * math.tan(self.camera_fov / 2))
        self.fy = self.image_height / (2 * math.tan(self.camera_fov / 2))
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        # Visualisation
        self.point_size = 6
        self.decay_time = 10.0
        
        self.bridge = CvBridge()
        self.image_couleur = None
        self.mask = None
        self.dernier_masque = None
        self.derniers_points = []
        self.compteur_perte = 0
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        
        self.distance = 0.0
        self.angle = 0.0
        
        # Subscribers
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_mask = self.create_subscription(Image, '/detected/mask', self.mask_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/lidar_2d', self.lidar_callback, 10)
        self.sub_dist = self.create_subscription(Float64, '/line_distance', self.distance_callback, 10)
        self.sub_angle = self.create_subscription(Float64, '/line_angle', self.angle_callback, 10)
        
        # Publishers
        self.pub_lines_3d = self.create_publisher(PointCloud2, '/lines_3d', 10)
        self.pub_fusion_image = self.create_publisher(Image, '/fusion/image', 10)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("FUSION NODE - AVEC CALIBRATION EXTRINSÈQUE")
        self.get_logger().info(f"📏 Translation: x={self.calib_x}, y={self.calib_y}, z={self.calib_z}")
        self.get_logger().info(f"🔄 Rotation: {math.degrees(self.calib_angle):.1f}°")
        self.get_logger().info(f"📷 Intrinsèque: fx={self.fx_intrinsic}, fy={self.fy_intrinsic}")
        self.get_logger().info(f"🎯 Offsets: u={self.offset_u}, v={self.offset_v}")
        self.get_logger().info("=" * 60)
    
    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(img, (self.image_width, self.image_height))
        except:
            pass
    
    def mask_callback(self, msg):
        try:
            mask_cv = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            kernel = np.ones((5,5), np.uint8)
            mask_cv = cv2.dilate(mask_cv, kernel, iterations=3)
            if np.sum(mask_cv) > 100:
                self.dernier_masque = mask_cv
                self.compteur_perte = 0
                self.mask = cv2.resize(mask_cv, (self.image_width, self.image_height))
            else:
                self.compteur_perte += 1
                if self.compteur_perte < 100 and self.dernier_masque is not None:
                    self.mask = cv2.resize(self.dernier_masque, (self.image_width, self.image_height))
                else:
                    self.mask = cv2.resize(mask_cv, (self.image_width, self.image_height))
        except:
            if self.dernier_masque is not None:
                self.mask = cv2.resize(self.dernier_masque, (self.image_width, self.image_height))
    
    def distance_callback(self, msg):
        self.distance = msg.data
    
    def angle_callback(self, msg):
        self.angle = msg.data
    
    def transform_lidar_to_camera(self, x, y, z):
        """Transforme un point du repère LiDAR vers le repère caméra"""
        pt_lidar = np.array([x, y, z, 1.0])
        pt_cam = self.T_lidar_camera @ pt_lidar
        return pt_cam[0], pt_cam[1], pt_cam[2]
    
    def project_to_image(self, x, y, z):
        """Projette un point 3D sur l'image"""
        if x <= 0.01:
            return None
        u = int(self.fx_intrinsic * (y / x) + self.cx_intrinsic + self.offset_u)
        v = int(self.fy_intrinsic * (z / x) + self.cy_intrinsic + self.offset_v)
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return (u, v)
        return None
    
    def lidar_callback(self, msg):
        if self.image_couleur is None or self.mask is None:
            return
        
        self.frame_count += 1
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = current_time
        
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if len(points) == 0:
            return
        
        points_sur_lignes = []
        for p in points:
            x, y, z, intensity = p
            if not (self.x_min < x < self.x_max and self.y_min < y < self.y_max):
                continue
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            if pixel is not None:
                u, v = pixel
                if self.mask[v, u] > 0:
                    points_sur_lignes.append([x, y, z, intensity])
        
        # Persistance
        if len(points_sur_lignes) == 0 and len(self.derniers_points) > 0:
            points_sur_lignes = self.derniers_points
        else:
            self.derniers_points = points_sur_lignes
        
        # Publication 3D
        if len(points_sur_lignes) > 0:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            cloud = pc2.create_cloud(msg.header, fields, points_sur_lignes)
            self.pub_lines_3d.publish(cloud)
        
        # Visualisation
        vis_image = self.image_couleur.copy()
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 4)
        
        for p in points_sur_lignes:
            x, y, z, intensity = p
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            if pixel is not None:
                u, v = pixel
                cv2.circle(vis_image, (u, v), self.point_size, (0, 0, 255), -1)
        
        # Informations
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Points sur ligne: {len(points_sur_lignes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Distance ligne: {self.distance:.2f} m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Angle ligne: {math.degrees(self.angle):.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Calib: x={self.calib_x:.2f}, y={self.calib_y:.2f}, z={self.calib_z:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        img_msg = self.bridge.cv2_to_imgmsg(vis_image, 'bgr8')
        self.pub_fusion_image.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
