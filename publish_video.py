#!/usr/bin/env python3
"""
Nœud pour publier une vidéo (MP4, AVI, etc.) sur /camera/image_raw
Avec le frame_id "camera_frame" pour la TF
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import os

class VideoPublisher(Node):
    def __init__(self, video_path, fps=15):
        super().__init__('video_publisher')
        
        # Publisher avec le bon topic
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # Vérifier que la vidéo existe
        if not os.path.exists(video_path):
            self.get_logger().error(f"❌ Vidéo non trouvée: {video_path}")
            return
        
        # Ouvrir la vidéo
        self.cap = cv2.VideoCapture(video_path)
        self.fps = fps
        
        # Obtenir les infos de la vidéo
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.get_logger().info(f"📹 Vidéo: {video_path}")
        self.get_logger().info(f"   Résolution: {width}x{height}")
        self.get_logger().info(f"   FPS vidéo: {video_fps:.1f}")
        self.get_logger().info(f"   Frames: {total_frames}")
        self.get_logger().info(f"   Publication FPS: {self.fps}")
        self.get_logger().info(f"   Topic: /camera/image_raw")
        self.get_logger().info(f"   frame_id: camera_frame")
        
        # Timer pour publier à la fréquence demandée
        self.timer = self.create_timer(1.0/self.fps, self.timer_callback)
        self.frame_count = 0
    
    def timer_callback(self):
        # Lire une frame
        ret, frame = self.cap.read()
        
        # Si fin de la vidéo, revenir au début
        if not ret:
            self.get_logger().info("🔄 Vidéo en boucle - retour au début")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        
        # Incrémenter le compteur
        self.frame_count += 1
        
        # Convertir OpenCV → ROS Image
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        
        # IMPORTANT : Ajouter le timestamp et le frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"   # ← Clé pour la TF !
        
        # Publier
        self.publisher.publish(msg)
        
        # Log toutes les 100 frames
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"📷 Frame {self.frame_count} publiée")

def main(args=None):
    if len(sys.argv) < 2:
        print("=" * 60)
        print("📹 PUBLISH VIDEO")
        print("=" * 60)
        print("Usage: ros2 run code_image publish_video <chemin_video> [fps]")
        print("")
        print("Exemples:")
        print("  ros2 run code_image publish_video /home/amin/Downloads/video.mp4")
        print("  ros2 run code_image publish_video /home/amin/Downloads/video.mp4 20")
        print("=" * 60)
        return
    
    video_path = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    rclpy.init(args=args)
    node = VideoPublisher(video_path, fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
