#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class FiltrageLidar3(Node):
    def __init__(self):
        super().__init__('filtrage_lidar3')
        
        # Paramètres de filtrage
        self.X_MIN = 0.0
        self.X_MAX = 0.5
        self.Y_MIN = -2.0
        self.Y_MAX = 2.0
        self.Z_MIN = 0.2
        self.Z_MAX = 5.0
        
        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10
        )
        
        # Publisher
        self.publisher = self.create_publisher(
            PointCloud2,
            '/lidar3/points_2d',
            10
        )
        
        self.get_logger().info("✅ Nœud filtrage_lidar3 démarré")
        self.get_logger().info(f"Filtre: X[{self.X_MIN},{self.X_MAX}] Y[{self.Y_MIN},{self.Y_MAX}] Z[{self.Z_MIN},{self.Z_MAX}]")
        self.get_logger().info("Publication sur /lidar3/points_2d")
        
        self.frame_count = 0
    
    def callback(self, msg):
        self.frame_count += 1
        
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if len(points) == 0:
            return
        
        points_2d = []
        
        for x, y, z in points:
            if (self.X_MIN < x < self.X_MAX and 
                self.Y_MIN < y < self.Y_MAX and 
                self.Z_MIN < z < self.Z_MAX):
                points_2d.append([x, y, 0.0])
        
        if len(points_2d) > 0:
            cloud_2d = pc2.create_cloud_xyz32(msg.header, points_2d)
            self.publisher.publish(cloud_2d)
        
        if self.frame_count % 50 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {len(points)} → {len(points_2d)} points 2D")

def main(args=None):
    rclpy.init(args=args)
    node = FiltrageLidar3()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
       
