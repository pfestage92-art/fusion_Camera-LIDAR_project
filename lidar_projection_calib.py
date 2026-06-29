#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarProjectionCalib(Node):
    def __init__(self):
        super().__init__('lidar_projection_calib')

        # Paramètres de base
        self.declare_parameter('point_size', 4)
        self.declare_parameter('decay_time', 5.0)
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        
        # Paramètres intrinsèques (matrice K)
        self.declare_parameter('fx', 2000.0)
        self.declare_parameter('fy', 2000.0)
        self.declare_parameter('cx', 1024.0)
        self.declare_parameter('cy', 768.0)
        
        # NOUVEAU : Paramètres de distorsion
        self.declare_parameter('k1', -0.0388035566)
        self.declare_parameter('k2', -17.1594362)
        self.declare_parameter('p1', 0.117463586)
        self.declare_parameter('p2', 0.0475253551)
        self.declare_parameter('k3', 113.175046)
        
        # Offsets (maintenant à 0 car la distorsion compense)
        self.declare_parameter('offset_u', 0)
        self.declare_parameter('offset_v', 0)

        # Récupération des paramètres
        self.point_size = self.get_parameter('point_size').value
        self.decay_time = self.get_parameter('decay_time').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        
        # Récupération des coefficients de distorsion
        self.k1 = self.get_parameter('k1').value
        self.k2 = self.get_parameter('k2').value
        self.p1 = self.get_parameter('p1').value
        self.p2 = self.get_parameter('p2').value
        self.k3 = self.get_parameter('k3').value
        
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value

        # Matrice extrinsèque (LiDAR → Caméra)
        self.T = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.18],
            [0.0, 0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.bridge = CvBridge()
        self.image_couleur = None
        self.mask = None
        self.points_accumules = []
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = self.get_clock().now()

        # Subscribers
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/unilidar/cloud', self.lidar_callback, 10)
        self.sub_mask = self.create_subscription(Image, '/detected/mask', self.mask_callback, 10)
        
        # Publishers
        self.pub = self.create_publisher(Image, '/lidar/projection', 10)
        self.pub_points = self.create_publisher(PointCloud2, '/lidar/projected_points', 10)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  LiDAR Projection AVEC distorsion")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"📷 K: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        self.get_logger().info(f"🔧 Distorsion: k1={self.k1}, k2={self.k2}, k3={self.k3}")
        self.get_logger().info(f"🔧 Distorsion: p1={self.p1}, p2={self.p2}")
        self.get_logger().info("=" * 55)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(img, (self.image_width, self.image_height))
        except:
            pass

    def mask_callback(self, msg):
        try:
            mask_cv = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.mask = cv2.resize(mask_cv, (self.image_width, self.image_height))
        except:
            pass

    def apply_distortion(self, x, y):
        """
        Applique la correction de distorsion sur des coordonnées normalisées (x, y)
        Retourne les coordonnées corrigées (x_dist, y_dist)
        """
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r2*r4
        
        # Correction radiale
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        
        # Correction tangentielle
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x*x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y*y) + 2 * self.p2 * x * y
        
        return x_dist, y_dist

    def get_point_color(self, depth, on_line):
        if on_line:
            return (0, 0, 255)  # Rouge pour les points sur les lignes
        dmin, dmax = 0.3, 8.0
        t = np.clip((depth - dmin) / (dmax - dmin), 0, 1)
        r = int((1 - t) * 255)
        g = int(math.sin(t * math.pi) * 255)
        b = int(t * 255)
        return (b, g, r)

    def lidar_callback(self, msg):
        if self.image_couleur is None:
            return

        self.frame_count += 1
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time.nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt
        self.last_time = self.get_clock().now()

        # Lecture des points LiDAR
        points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
        if not points:
            return

        # Transformation des points en tableau numpy
        pts = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float64)
        n = len(pts)

        # Transformation LiDAR → Caméra (extrinsèque)
        pts_h = np.hstack([pts, np.ones((n, 1))])
        pts_cam = (self.T @ pts_h.T).T[:, :3]
        
        # Filtrage : garder les points devant la caméra (Z > 0.1m)
        mask_z = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask_z]
        if len(pts_cam) == 0:
            return

        # ===== NOUVEAU : Projection AVEC distorsion =====
        # Étape 1 : Coordonnées normalisées
        x_proj = pts_cam[:, 0] / pts_cam[:, 2]
        y_proj = pts_cam[:, 1] / pts_cam[:, 2]
        
        # Étape 2 : Application de la distorsion (pour chaque point)
        x_dist = np.zeros_like(x_proj)
        y_dist = np.zeros_like(y_proj)
        
        for i in range(len(x_proj)):
            xd, yd = self.apply_distortion(x_proj[i], y_proj[i])
            x_dist[i] = xd
            y_dist[i] = yd
        
        # Étape 3 : Projection avec matrice K (plus besoin d'offsets)
        u = (self.fx * x_dist + self.cx + self.offset_u).astype(int)
        v = (self.fy * y_dist + self.cy + self.offset_v).astype(int)
        d = pts_cam[:, 2]
        # ================================================

        # Filtrage : garder les points dans l'image
        valid = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, d = u[valid], v[valid], d[valid]

        # Publication des points projetés pour le distance_estimator
        points_3d = []
        for i in range(len(u)):
            points_3d.append([float(u[i]), float(v[i]), float(d[i]), 100.0])
        
        if points_3d:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            cloud_msg = pc2.create_cloud(msg.header, fields, points_3d)
            self.pub_points.publish(cloud_msg)

        # Association avec les masques et coloration
        nouveaux = []
        for i in range(len(u)):
            on_line = False
            if self.mask is not None and self.mask[v[i], u[i]] > 0:
                on_line = True
            color = self.get_point_color(d[i], on_line)
            nouveaux.append((int(u[i]), int(v[i]), color, current_time))
        
        # Ajout des nouveaux points
        self.points_accumules.extend(nouveaux)
        
        # Suppression des points trop anciens (persistance temporelle)
        self.points_accumules = [p for p in self.points_accumules if current_time - p[3] < self.decay_time]

        # Visualisation
        vis = self.image_couleur.copy()
        
        # Dessiner les contours du masque
        if self.mask is not None:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Dessiner les points accumulés
        points_sur_lignes = 0
        for pu, pv, color, _ in self.points_accumules:
            cv2.circle(vis, (pu, pv), self.point_size, color, -1)
            if color == (0, 0, 255):
                points_sur_lignes += 1

        # Affichage des informations
        cv2.putText(vis, f"Points: {len(self.points_accumules)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, f"Points ROUGES: {points_sur_lignes}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"FPS: {self.fps:.1f}", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(vis, "Avec distorsion", (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Publication de l'image fusionnée
        self.pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = LidarProjectionCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé par l'utilisateur")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
