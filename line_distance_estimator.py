#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float64
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from collections import deque

class DistanceEstimator(Node):
    def __init__(self):
        super().__init__('distance_estimator')
        
        self.cx = 1024
        self.cy = 768
        
        # Coefficient calibré (à ajuster)
        self.coeff = 200  # Augmenté pour des distances plus petites
        
        # Filtre moyenne mobile sur 10 échantillons
        self.hist_gauche = deque(maxlen=10)
        self.hist_droite = deque(maxlen=10)
        
        self.bridge = CvBridge()
        self.image = None
        
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_cb, 10)
        self.sub_points = self.create_subscription(PointCloud2, '/lidar/projected_points', self.points_cb, 10)
        
        self.pub_result = self.create_publisher(Image, '/distance/result', 10)
        self.pub_gauche = self.create_publisher(Float64, '/distance/gauche', 10)
        self.pub_droite = self.create_publisher(Float64, '/distance/droite', 10)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("Distance Estimator - Version stabilisée")
        self.get_logger().info(f"Coefficient = {self.coeff}")
        self.get_logger().info("=" * 50)
    
    def image_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def pixel_to_distance(self, v):
        delta_v = abs(v - self.cy)
        if delta_v < 5:
            delta_v = 5
        return self.coeff / delta_v
    
    def smooth(self, value, hist):
        hist.append(value)
        return np.mean(hist)
    
    def points_cb(self, msg):
        if self.image is None:
            return
        
        points_gauche = []
        points_droite = []
        
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            u, v = int(p[0]), int(p[1])
            if 0 <= u < 2048 and 0 <= v < 1536:
                if u < self.cx - 80:
                    points_gauche.append((u, v))
                elif u > self.cx + 80:
                    points_droite.append((u, v))
        
        vis = self.image.copy()
        
        # === LIGNE GAUCHE ===
        dist_gauche = 0.0
        if len(points_gauche) >= 5:
            X = np.array([[u] for (u, v) in points_gauche])
            y = np.array([v for (u, v) in points_gauche])
            ransac = RANSACRegressor(residual_threshold=15, random_state=42)
            ransac.fit(X, y)
            a, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            
            v_center = int(a * self.cx + b)
            dist_raw = self.pixel_to_distance(v_center)
            dist_gauche = self.smooth(dist_raw, self.hist_gauche)
            
            y1, y2 = int(a * 0 + b), int(a * 2048 + b)
            cv2.line(vis, (0, y1), (2048, y2), (0, 0, 0), 4)
            for u, v in points_gauche:
                cv2.circle(vis, (u, v), 3, (0, 0, 0), -1)
        
        # === LIGNE DROITE ===
        dist_droite = 0.0
        if len(points_droite) >= 5:
            X = np.array([[u] for (u, v) in points_droite])
            y = np.array([v for (u, v) in points_droite])
            ransac = RANSACRegressor(residual_threshold=15, random_state=42)
            ransac.fit(X, y)
            a, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            
            v_center = int(a * self.cx + b)
            dist_raw = self.pixel_to_distance(v_center)
            dist_droite = self.smooth(dist_raw, self.hist_droite)
            
            y1, y2 = int(a * 0 + b), int(a * 2048 + b)
            cv2.line(vis, (0, y1), (2048, y2), (255, 0, 0), 4)
            for u, v in points_droite:
                cv2.circle(vis, (u, v), 3, (255, 0, 0), -1)
        
        # Affichage
        cv2.putText(vis, f"GAUCHE: {dist_gauche:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(vis, f"DROITE: {dist_droite:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.circle(vis, (self.cx, self.cy), 10, (0, 255, 0), -1)
        
        self.pub_gauche.publish(Float64(data=float(dist_gauche)))
        self.pub_droite.publish(Float64(data=float(dist_droite)))
        self.pub_result.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        
        self.get_logger().info(f"📏 GAUCHE: {dist_gauche:.3f} m | DROITE: {dist_droite:.3f} m")

def main(args=None):
    rclpy.init(args=args)
    node = DistanceEstimator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
