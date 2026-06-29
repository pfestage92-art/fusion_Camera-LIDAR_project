#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

class LidarTFTransformer(Node):
    def __init__(self):
        super().__init__('lidar_tf_transformer')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.sub = self.create_subscription(PointCloud2, '/unilidar/cloud', self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, '/lidar_in_camera_frame', 10)
        
        self.get_logger().info("Lidar TF Transformer - publie les points dans le repère camera_frame")
    
    def callback(self, msg):
        try:
            # Lire tous les points
            points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            transformed_points = []
            
            for p in points:
                # Créer un point dans le repère LiDAR
                ps = PointStamped()
                ps.header.frame_id = "unilidar_lidar"
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.point.x = float(p[0])
                ps.point.y = float(p[1])
                ps.point.z = float(p[2])
                
                # Transformer dans le repère caméra
                ps_transformed = self.tf_buffer.transform(ps, "camera_frame", timeout=rclpy.duration.Duration(seconds=0.1))
                transformed_points.append([ps_transformed.point.x, ps_transformed.point.y, ps_transformed.point.z])
            
            # Publier le nuage transformé
            if transformed_points:
                cloud_out = pc2.create_cloud_xyz32(msg.header, transformed_points)
                cloud_out.header.frame_id = "camera_frame"
                self.pub.publish(cloud_out)
                
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarTFTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
