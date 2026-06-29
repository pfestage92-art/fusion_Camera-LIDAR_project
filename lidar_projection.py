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

        # ── Paramètres image ──────────────────────────
        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('point_size',   3)
        self.declare_parameter('decay_time',   5.0)

        # ── Intrinsèques caméra ───────────────────────
        self.declare_parameter('fx', 800.0)
        self.declare_parameter('fy', 800.0)
        self.declare_parameter('cx', 370.0)   # image_width/2 + 50
        self.declare_parameter('cy', 170.0)   # image_height/2 - 70

        # ── Extrinsèques LiDAR → Caméra ──────────────
        # Translation (mètres) : position du LiDAR par rapport à la caméra
        self.declare_parameter('tx',  0.0)    # gauche/droite
        self.declare_parameter('ty', -0.08)   # haut/bas (LiDAR au-dessus → négatif)
        self.declare_parameter('tz',  0.0)    # avant/arrière

        # Rotation (degrés)
        self.declare_parameter('rx',  0.0)    # pitch
        self.declare_parameter('ry',  0.0)    # yaw
        self.declare_parameter('rz',  0.0)    # roll

        # ── Lecture paramètres ────────────────────────
        self.image_width  = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.point_size   = self.get_parameter('point_size').value
        self.decay_time   = self.get_parameter('decay_time').value

        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        tx = self.get_parameter('tx').value
        ty = self.get_parameter('ty').value
        tz = self.get_parameter('tz').value
        rx = math.radians(self.get_parameter('rx').value)
        ry = math.radians(self.get_parameter('ry').value)
        rz = math.radians(self.get_parameter('rz').value)

        # ── Matrice de transformation 4x4 ────────────
        Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
        Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
        Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
        R  = Rz @ Ry @ Rx

        self.T = np.eye(4)
        self.T[:3,:3] = R
        self.T[:3, 3] = [tx, ty, tz]

        # ── Matrice intrinsèque K ─────────────────────
        self.K = np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1]
        ])

        # ── Variables ─────────────────────────────────
        self.bridge           = CvBridge()
        self.image_couleur    = None
        self.points_accumules = []
        self.frame_count      = 0
        self.fps              = 0.0
        self.last_time        = self.get_clock().now()

        # ── Topics ────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/unilidar/cloud', self.lidar_callback, 10)
        self.pub = self.create_publisher(Image, '/lidar/projection', 10)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  LiDAR Projection Node démarré")
        self.get_logger().info(f"  T =\n{self.T}")
        self.get_logger().info(f"  K = fx={self.fx} fy={self.fy} cx={self.cx} cy={self.cy}")
        self.get_logger().info("=" * 55)

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(img, (self.image_width, self.image_height))
        except Exception as e:
            self.get_logger().warn(f"Image error: {e}")

    def get_color_depth(self, depth, dmin=0.3, dmax=8.0):
        """Couleur basée sur la profondeur : rouge=proche, bleu=loin"""
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

        # FPS
        dt = current_time - self.last_time.nanoseconds / 1e9
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt
        self.last_time = self.get_clock().now()

        # Lecture points LiDAR
        points = list(pc2.read_points(
            msg, field_names=("x","y","z","intensity"), skip_nans=True))
        if not points:
            return

        pts = np.array([(p[0],p[1],p[2]) for p in points], dtype=np.float64)
        n   = len(pts)

        # ── Transformation LiDAR → Caméra ────────────
        pts_h   = np.hstack([pts, np.ones((n,1))])      # (N,4)
        pts_cam = (self.T @ pts_h.T).T[:,:3]            # (N,3)

        # Garde seulement les points devant la caméra
        mask    = pts_cam[:,2] > 0.1
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            return

        # ── Projection perspective ────────────────────
        u = (self.fx * pts_cam[:,0] / pts_cam[:,2] + self.cx).astype(int)
        v = (self.fy * pts_cam[:,1] / pts_cam[:,2] + self.cy).astype(int)
        d = pts_cam[:,2]

        valid = (u>=0)&(u<self.image_width)&(v>=0)&(v<self.image_height)
        u, v, d = u[valid], v[valid], d[valid]

        # Accumulation avec timestamp
        nouveaux = [(int(u[i]), int(v[i]),
                     self.get_color_depth(d[i]), current_time)
                    for i in range(len(u))]
        self.points_accumules.extend(nouveaux)
        self.points_accumules = [
            p for p in self.points_accumules
            if current_time - p[3] < self.decay_time
        ]

        # ── Dessin ───────────────────────────────────
        vis = self.image_couleur.copy()
        for pu, pv, color, _ in self.points_accumules:
            cv2.circle(vis, (pu, pv), self.point_size, color, -1)

        cv2.putText(vis, f"Points: {len(self.points_accumules)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(vis, f"FPS: {self.fps:.1f}",
                    (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(vis, f"Projected: {len(u)}/{n}",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        self.pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {len(u)}/{n} pts projetés | FPS: {self.fps:.1f}")


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
