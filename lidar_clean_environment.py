#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class LidarCleanEnvironment(Node):
    def __init__(self):
        super().__init__('lidar_clean_environment')
        
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('camera_fov', 80.0)
        
        self.declare_parameter('x_min', 0.3)
        self.declare_parameter('x_max', 20.0)
        self.declare_parameter('y_min', -6.0)
        self.declare_parameter('y_max', 6.0)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 2.5)
        
        self.declare_parameter('calib_x', 0.215)
        self.declare_parameter('calib_y', -0.06)
        self.declare_parameter('calib_z', 0.17)
        self.declare_parameter('calib_angle', 10.0)
        
        self.declare_parameter('rotation_angle', 0.0)
        self.declare_parameter('invert_y', True)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('decay_time', 3.0)
        self.declare_parameter('point_size', 3)
        
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.camera_fov = math.radians(self.get_parameter('camera_fov').value)
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.calib_x = self.get_parameter('calib_x').value
        self.calib_y = self.get_parameter('calib_y').value
        self.calib_z = self.get_parameter('calib_z').value
        self.calib_angle = math.radians(self.get_parameter('calib_angle').value)
        self.rotation_angle = math.radians(self.get_parameter('rotation_angle').value)
        self.invert_y = self.get_parameter('invert_y').value
        self.invert_x = self.get_parameter('invert_x').value
        self.decay_time = self.get_parameter('decay_time').value
        self.point_size = self.get_parameter('point_size').value
        
        self.fx = self.image_width / (2 * math.tan(self.camera_fov / 2))
        self.fy = self.image_height / (2 * math.tan(self.camera_fov / 2))
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        
        self.bridge = CvBridge()
        self.image_couleur = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        self.points_accumules = []
        
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, '/unilidar/cloud', self.lidar_callback, 10)
        self.pub_clean = self.create_publisher(Image, '/lidar/clean_environment', 10)
        
        self.get_logger().info("LIDAR CLEAN ENVIRONMENT - AVEC INVERSION Y")
    
    def image_callback(self, msg):
        try:
            self.image_couleur = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_couleur = cv2.resize(self.image_couleur, (self.image_width, self.image_height))
        except:
            pass
    
    def transform_point(self, x, y):
        if self.invert_y:
            y = -y
        if self.invert_x:
            x = -x
        cos_a = math.cos(self.rotation_angle)
        sin_a = math.sin(self.rotation_angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        return x_rot, y_rot
    
    def transform_lidar_to_camera(self, x, y, z):
        x, y = self.transform_point(x, y)
        x_t = x - self.calib_x
        y_t = y - self.calib_y
        z_t = z - self.calib_z
        cos_a = math.cos(self.calib_angle)
        sin_a = math.sin(self.calib_angle)
        x_cam = x_t * cos_a - y_t * sin_a
        y_cam = x_t * sin_a + y_t * cos_a
        z_cam = z_t
        return x_cam, y_cam, z_cam
    
    def project_to_image(self, x, y, z):
        if x <= 0.01:
            return None
        u = int(self.fx * (y / x) + self.cx)
        v = int(self.fy * (z / x) + self.cy)
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return (u, v)
        return None
    
    def get_color(self, x, y, z):
        distance = math.sqrt(x*x + y*y + z*z)
        if -0.2 < z < 0.2:
            return (0, 0, 255)
        if distance < 1.5:
            return (0, 0, 255)
        elif distance < 3.0:
            return (0, 100, 255)
        elif distance < 6.0:
            return (0, 255, 255)
        elif distance < 10.0:
            return (0, 255, 0)
        else:
            return (128, 128, 128)
    
    def lidar_callback(self, msg):
        if self.image_couleur is None:
            return
        
        self.frame_count += 1
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        dt = (current_time - self.last_time.nanoseconds / 1e9) if hasattr(self.last_time, 'nanoseconds') else 0.1
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = self.get_clock().now()
        
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        
        if len(points) == 0:
            return
        
        nouveaux_points = []
        points_projetes = 0
        
        for p in points:
            x, y, z, intensity = p
            if not (self.x_min < x < self.x_max):
                continue
            if not (self.y_min < y < self.y_max):
                continue
            if not (self.z_min < z < self.z_max):
                continue
            
            x_cam, y_cam, z_cam = self.transform_lidar_to_camera(x, y, z)
            pixel = self.project_to_image(x_cam, y_cam, z_cam)
            
            if pixel is not None:
                u, v = pixel
                points_projetes += 1
                color = self.get_color(x, y, z)
                nouveaux_points.append((u, v, color, current_time))
        
        self.points_accumules.extend(nouveaux_points)
        self.points_accumules = [(u, v, c, t) for (u, v, c, t) in self.points_accumules 
                                  if current_time - t < self.decay_time]
        
        vis_image = self.image_couleur.copy()
        
        for u, v, color, _ in self.points_accumules:
            cv2.circle(vis_image, (u, v), self.point_size, color, -1)
        
        cv2.putText(vis_image, "LIGNES: ROUGE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(vis_image, f"Points: {len(self.points_accumules)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis_image, f"FPS: {self.fps:.1f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        img_msg = self.bridge.cv2_to_imgmsg(vis_image, 'bgr8')
        self.pub_clean.publish(img_msg)
        
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"Frame {self.frame_count}: {points_projetes} points | FPS: {self.fps:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarCleanEnvironment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
