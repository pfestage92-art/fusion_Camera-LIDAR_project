#!/usr/bin/env python3
"""
NŒUD DE DÉTECTION DE LIGNES AVEC YOLO - SEUIL À 90%
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import time

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # PARAMÈTRES
        self.declare_parameter('model_path', '/home/amin/ws/src/code_image/train14/weights/best.pt')
        self.declare_parameter('conf_threshold', 0.90)      # SEUIL À 90%
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/detected/image')
        self.declare_parameter('mask_topic', '/detected/mask')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('max_det', 100)
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.mask_topic = self.get_parameter('mask_topic').value
        self.imgsz = self.get_parameter('imgsz').value
        self.iou = self.get_parameter('iou').value
        self.max_det = self.get_parameter('max_det').value
        
        # INITIALISATION
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Modèle non trouvé: {model_path}")
            return
        
        self.get_logger().info(f"📦 Chargement du modèle: {model_path}")
        self.model = YOLO(model_path)
        self.get_logger().info("✅ Modèle chargé")
        
        self.bridge = CvBridge()
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        
        # Publishers & Subscribers
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)
        self.subscription = self.create_subscription(Image, input_topic, self.callback, 10)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"📷 DÉTECTION - Seuil: {self.conf_threshold*100}%")
        self.get_logger().info("=" * 50)
    
    def callback(self, msg):
        try:
            start_time = time.time()
            
            # Conversion image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
            
            # Détection YOLO avec seuil à 90%
            results = self.model(
                cv_image, 
                conf=self.conf_threshold,  # 0.90 = 90%
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False
            )
            
            # Image annotée
            annotated_image = results[0].plot(boxes=False)
            
            # Génération du masque
            h, w = cv_image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            nb_lignes = 0
            
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                nb_lignes = len(masks)
                
                for mask in masks:
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                
                # Nettoyer le masque
                kernel = np.ones((3,3), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Informations sur l'image (seuil à 90%)
            cv2.putText(annotated_image, f"Seuil: {self.conf_threshold*100}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_image, f"Lignes detectees: {nb_lignes}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Publications
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
            
            if combined_mask is not None and nb_lignes > 0:
                mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, 'mono8')
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)
            
            # FPS
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_time = current_time
            
            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f"Frame {self.frame_count}: {nb_lignes} lignes (conf > 90%) | "
                    f"Temps: {inference_time:.0f}ms | FPS: {self.fps:.1f}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Erreur: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
