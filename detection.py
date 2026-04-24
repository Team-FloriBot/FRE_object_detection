from ultralytics import YOLO
import cv2
import os
import numpy as np
#import pyrealsense2 as rs
import time
from pathlib import Path
OUT_IMG = "dataset/images/test"
OUT_IMG = "debug_images"

"""
IMG = Path("dataset/images/train/synth_0.jpg")   # eine deiner Dateien
LBL = Path("dataset/labels/train/synth_0.txt")   # passende Labeldatei

img = cv2.imread(str(IMG))
h, w = img.shape[:2]

with LBL.open() as f:
    for line in f:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        poly = np.array(coords, dtype=float).reshape(-1, 2)
        poly[:, 0] *= w
        poly[:, 1] *= h
        pts = poly.astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, str(cls), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

cv2.imwrite(os.path.join(OUT_IMG, "new1.jpg"), img)
cv2.waitKey(0)"""


class ObjDetection:
    def __init__(self, classes):
        self.W=640
        self.H=480

        # Initialize a YOLOE model
        self.model = YOLO("/model/yolo26n_jute_stripe_yellow_paper-seg.pt")
        # Save classes to detect
        self.classes = classes
        self.class_ids = [id for id in self.model.names if self.model.names[id] in classes]

        # Initialize webcam
        #self.cap = cv2.VideoCapture(0)

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self):
        # init realsense
        pass


    # ---------------------------------------------------------
    # Capture Frame in camera
    # ---------------------------------------------------------
    def get_frame(self, out_name):
        """""""""""""""""""""""""""
        RGB- und Tiefenbild lesen
        Outputs: color_image, depth_image
        """""""""""""""""""""""""""
        
        # --> replace that with rs
        #ret, color_image = self.cap.read()
        color_image = cv2.imread(os.path.join(OUT_IMG, out_name))
        


        # --> replace that with rs
        #color_image = cv2.resize(color_image, (self.W, self.H))
        depth_image = None
        return color_image, depth_image


    # ---------------------------------------------------------
    # Run Detection
        # ---------------------------------------------------------
    def detect_obj(self, color_image):
        color_image = cv2.resize(color_image, (self.W, self.H))

        results = self.model.predict(
            color_image,
            classes=self.class_ids,
            conf=0.25,
            imgsz=640,
            rect=True,
        )

        annotated_image = color_image.copy()
        frame_masks = []
        result = results[0]

        # Falls YOLO keine Masken erzeugt
        if result.masks is None:
            return frame_masks, annotated_image

        import torch
        import torch.nn.functional as F

        orig_h, orig_w = color_image.shape[:2]

        # Iteriere über alle Detektionen
        for box, mask_tensor in zip(result.boxes, result.masks.data):

            # 1) Byte → Float konvertieren (Pflicht!)
            mask_tensor = mask_tensor.float()

            # 2) Auf Originalgröße skalieren
            full_mask = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()

            # 3) Binarisieren
            mask_uint8 = ((full_mask > 0.5) * 255).astype("uint8")

            # 4) Overlay erzeugen
            color_mask = np.zeros_like(annotated_image)
            color_mask[mask_uint8 > 0] = [255, 0, 0]  # Blau

            annotated_image = cv2.addWeighted(
                annotated_image, 1,
                color_mask, 0.5,
                0
            )

            # 5) Ergebnis speichern
            frame_masks.append({
                "class": self.model.names[int(box.cls[0])],
                "mask": mask_uint8
            })

        return frame_masks, annotated_image


    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------
    def fuse(self, color_image, depth_image, frame_masks):
        """""""""""""""""""""""""""
        calculate coordinates of detected images
        Input color_image, depth_image, frame_masks
        Output coordinates_results --> {label1: [[x1_1,y1_1,z1_1], [x1_2,y1_2,z1_2]], label2: [[x2_1,y2_1,z2_1]]}
        """""""""""""""""""""""""""
        pass

    # ---------------------------------------------------------
    # Stop Camera
    # ---------------------------------------------------------
    def stop_camera(self):
        #self.cap.release()
        #cv2.destroyAllWindows()
        pass

    def run(self):
        for out_name in os.listdir(OUT_IMG):
            # Kamerabild lesen
            color_image, depth_image = self.get_frame(out_name)

            # Objekte detektieren
            frame_masks, annotated_color_image = self.detect_obj(color_image)

            # RGB- und Tiefenbild fussionieren
            #coordinates = self.fuse(color_image, depth_image, frame_masks)
            # Objekt-Koordinaten berechnen

            # Detektion anzeigen
            #cv2.imshow("Orginal", color_image)
            #cv2.imshow("Detektion", annotated_color_image)
            cv2.imwrite(os.path.join(OUT_IMG, f"pred_{out_name}"), annotated_color_image)
            

            # Beenden mit 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()

if __name__ == "__main__":
    person_detection = ObjDetection(["jute-stripe","yellow-paper"])
    person_detection.run()