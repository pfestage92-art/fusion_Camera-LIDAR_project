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

        # Déclaration des paramètres
        self.declare_parameter('point_size', 4)
        self.declare_parameter('decay_time', 5.0)
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        self.declare_parameter('fx', 2000.0)
        self.declare_parameter('fy', 2000.0)
        self.declare_parameter('cx', 1024.0)
        self.declare_parameter('cy', 768.0)
        self.declare_parameter('offset_u', 50)
        self.declare_parameter('offset_v', -200)

        # Récupération des paramètres
        self.point_size = self.get_parameter('point_size').value
        self.decay_time = self.get_parameter('decay_time').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.offset_u = self.get_parameter('offset_u').value
        self.offset_v = self.get_parameter('offset_v').value

        # Matrice extrinsèque
        self.T = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.18],
            [0.0, 0.0, 1.0, 0.70],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.bridge = CvBridge()
        self.image_couleur = None
        self.mask = None  # Masque des lignes détectées
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

        self.get_logger().info("=" * 55)
        self.get_logger().info("  LiDAR Projection — POINTS ROUGES SUR LIGNES")
        self.get_logger().info(f"  fx={self.fx} fy={self.fy} cx={self.cx} cy={self.cy}")
        self.get_logger().info(f"  offsets: u={self.offset_u} v={self.offset_v}")
        self.get_logger().info("=" * 55)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(img, (self.image_width, self.image_height))
        except:
            pass

    def mask_callback(self, msg):
        """Reçoit le masque des lignes détectées"""
        try:
            mask_cv = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.mask = cv2.resize(mask_cv, (self.image_width, self.image_height))
        except:
            pass

    def get_point_color(self, depth, on_line):
        """Retourne la couleur du point : ROUGE si sur ligne, sinon couleur selon distance"""
        if on_line:
            return (0, 0, 255)  # ROUGE
        # Couleur selon distance
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

        points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
        if not points:
            return

        pts = np.array([(p[0],p[1],p[2]) for p in points], dtype=np.float64)
        n = len(pts)

        pts_h = np.hstack([pts, np.ones((n,1))])
        pts_cam = (self.T @ pts_h.T).T[:,:3]
        mask_z = pts_cam[:,2] > 0.1
        pts_cam = pts_cam[mask_z]
        if len(pts_cam) == 0:
            return

        u = (self.fx * pts_cam[:,0] / pts_cam[:,2] + self.cx + self.offset_u).astype(int)
        v = (self.fy * pts_cam[:,1] / pts_cam[:,2] + self.cy + self.offset_v).astype(int)
        d = pts_cam[:,2]

        valid = (u>=0)&(u<self.image_width)&(v>=0)&(v<self.image_height)
        u, v, d = u[valid], v[valid], d[valid]

        # ============================================
        # DESSIN DES POINTS (ROUGES si sur lignes)
        # ============================================
        nouveaux = []
        for i in range(len(u)):
            # Vérifier si le point est sur une ligne détectée
            on_line = False
            if self.mask is not None and self.mask[v[i], u[i]] > 0:
                on_line = True
            
            color = self.get_point_color(d[i], on_line)
            nouveaux.append((int(u[i]), int(v[i]), color, current_time))
        
        self.points_accumules.extend(nouveaux)
        self.points_accumules = [p for p in self.points_accumules if current_time - p[3] < self.decay_time]

        # ============================================
        # VISUALISATION
        # ============================================
        vis = self.image_couleur.copy()
        
        # Dessiner le masque des lignes (contours verts)
        if self.mask is not None:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Dessiner les points
        points_sur_lignes = 0
        for pu, pv, color, _ in self.points_accumules:
            cv2.circle(vis, (pu, pv), self.point_size, color, -1)
            if color == (0, 0, 255):
                points_sur_lignes += 1

        # Informations
        cv2.putText(vis, f"Points: {len(self.points_accumules)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(vis, f"Points ROUGES (sur lignes): {points_sur_lignes}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(vis, f"FPS: {self.fps:.1f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

        self.pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = LidarProjectionCalib()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
