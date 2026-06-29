#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

class Lidar2D(Node):
    def __init__(self):
        super().__init__('lidar_2d_projection')
        
        # Subscriber - écoute le LiDAR 3D
        self.subscription = self.create_subscription(
            PointCloud2,
            '/unilidar/cloud',
            self.callback,
            10)
        
        # Publisher - publie les points projetés en 2D
        self.publisher = self.create_publisher(PointCloud2, '/lidar_2d', 10)
        
        # Compteur
        self.frame_count = 0
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("✅ NŒUD LIDAR 2D PROJECTION DÉMARRÉ")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Filtre: X ∈ [-1 , 2], Y ∈ [-1, 1], Z ∈ [0.2, 5]")
        self.get_logger().info("Projection: plan YZ (z, -y, 0)")
        self.get_logger().info("Publication sur: /lidar_2d")
        self.get_logger().info("=" * 50)

    def callback(self, msg):
        self.frame_count += 1
        
        # Lire les points LiDAR
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        
        if len(points) == 0:
            return
        
        projected_points = []
        
        for p in points:
            x, y, z, intensity = p


            
            # Filtre spatial
            if 0.6< x < 1 and -1 < y < 1 and 0.2 < z < 5:
                # Projection sur le plan YZ
                # z devient x, -y devient y, z=0
                projected_points.append([z, -y, 0.0, intensity])
        
        # Créer le message PointCloud2 avec intensity
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        if len(projected_points) > 0:
            cloud = pc2.create_cloud(msg.header, fields, projected_points)
            self.publisher.publish(cloud)
        
        # Afficher les stats toutes les 50 frames
        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: "
                f"{len(points)} points → "
                f"{len(projected_points)} points projetés"
            )

def main(args=None):
    rclpy.init(args=args)
    node = Lidar2D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
