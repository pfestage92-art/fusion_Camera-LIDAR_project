#!/usr/bin/env python3
"""
NŒUD DE DÉTECTION DE LIGNES AVEC YOLO - VERSION AMÉLIORÉE
Avec filtre temporel et optimisation temps réel
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
        
        # =============================================
        # PARAMÈTRES
        # =============================================
        self.declare_parameter('model_path', '/home/amin/ws/src/code_image/train14/weights/best.pt')
        self.declare_parameter('conf_threshold', 0.04)      # Seuil bas pour plus de détections
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/detected/image')
        self.declare_parameter('mask_topic', '/detected/mask')
        self.declare_parameter('imgsz', 640)                # Taille d'entrée YOLO
        self.declare_parameter('iou', 0.45)                 # IoU pour NMS
        self.declare_parameter('max_det', 100)              # Max détections par image
        self.declare_parameter('hist_len', 5)               # Longueur filtre temporel
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.mask_topic = self.get_parameter('mask_topic').value
        self.imgsz = self.get_parameter('imgsz').value
        self.iou = self.get_parameter('iou').value
        self.max_det = self.get_parameter('max_det').value
        self.hist_len = self.get_parameter('hist_len').value
        
        # =============================================
        # FILTRE TEMPOREL
        # =============================================
        self.historique_lignes = []      # Historique du nombre de lignes
        self.historique_masques = []     # Historique des masques
        self.historique_images = []      # Historique des images annotées
        
        # =============================================
        # INITIALISATION
        # =============================================
        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Modèle non trouvé: {model_path}")
            return
        
        self.get_logger().info(f"📦 Chargement du modèle: {model_path}")
        self.model = YOLO(model_path)
        self.get_logger().info("✅ Modèle chargé avec succès")
        
        self.bridge = CvBridge()
        self.frame_count = 0
        self.fps = 0
        self.last_time = self.get_clock().now()
        
        # =============================================
        # PUBLISHERS & SUBSCRIBERS
        # =============================================
        self.image_pub = self.create_publisher(Image, output_topic, 10)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)
        self.subscription = self.create_subscription(Image, input_topic, self.callback, 10)
        
        # =============================================
        # LOGS
        # =============================================
        self.get_logger().info("=" * 50)
        self.get_logger().info("📷 NŒUD DE DÉTECTION AMÉLIORÉ")
        self.get_logger().info(f"   Seuil: {self.conf_threshold}")
        self.get_logger().info(f"   Taille image: {self.imgsz}x{self.imgsz}")
        self.get_logger().info(f"   Filtre temporel: {self.hist_len} frames")
        self.get_logger().info("=" * 50)
    
    def lissage_temporel(self, masque, nb_lignes):
        """Applique un filtre temporel pour stabiliser les détections"""
        
        # Lisser le nombre de lignes
        self.historique_lignes.append(nb_lignes)
        if len(self.historique_lignes) > self.hist_len:
            self.historique_lignes.pop(0)
        nb_lignes_lisse = int(sum(self.historique_lignes) / len(self.historique_lignes))
        
        # Lisser le masque (optionnel - garde le masque le plus récent)
        self.historique_masques.append(masque)
        if len(self.historique_masques) > self.hist_len:
            self.historique_masques.pop(0)
        
        # Si détection instable, garder le masque précédent
        if nb_lignes == 0 and nb_lignes_lisse > 0 and len(self.historique_masques) > 1:
            masque = self.historique_masques[-2]
        
        return masque, nb_lignes_lisse
    
    def callback(self, msg):
        try:
            start_time = time.time()
            
            # Convertir ROS Image → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
            
            # =========================================
            # PRÉ-TRAITEMENT (améliore la détection)
            # =========================================
            # Améliorer le contraste
            cv_image = cv2.convertScaleAbs(cv_image, alpha=1.1, beta=5)
            
            # =========================================
            # DÉTECTION YOLO
            # =========================================
            results = self.model(
                cv_image, 
                conf=self.conf_threshold,
                iou=self.iou,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False
            )
            
            # =========================================
            # GÉNÉRATION DE L'IMAGE ANNOTÉE
            # =========================================
            annotated_image = results[0].plot(boxes=False)
            
            # =========================================
            # GÉNÉRATION DU MASQUE BINAIRE
            # =========================================
            masque = None
            nb_lignes = 0
            
            if results[0].masks is not None:
                h, w = cv_image.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                masks = results[0].masks.data.cpu().numpy()
                nb_lignes = len(masks)
                
                for mask in masks:
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                
                masque = combined_mask
                
                # =========================================
                # AMÉLIORATION DU MASQUE
                # =========================================
                # Nettoyer le masque (enlever le bruit)
                kernel = np.ones((3,3), np.uint8)
                masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, kernel)
                masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
                
                # =========================================
                # FILTRE TEMPOREL
                # =========================================
                masque, nb_lignes_lisse = self.lissage_temporel(masque, nb_lignes)
            else:
                # Mettre à jour l'historique avec 0
                self.lissage_temporel(np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.uint8), 0)
                nb_lignes_lisse = int(sum(self.historique_lignes) / len(self.historique_lignes)) if self.historique_lignes else 0
            
            # =========================================
            # PUBLICATION
            # =========================================
            # Publier l'image annotée
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
            
            # Publier le masque
            if masque is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(masque, 'mono8')
                mask_msg.header = msg.header
                self.mask_pub.publish(mask_msg)
            
            # =========================================
            # CALCUL DES FPS
            # =========================================
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_time = current_time
            
            # =========================================
            # LOGS
            # =========================================
            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f"Frame {self.frame_count}: {nb_lignes} lignes | "
                    f"Temps: {inference_time:.0f}ms | "
                    f"FPS: {self.fps:.1f}"
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
