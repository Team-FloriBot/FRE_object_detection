from ultralytics import YOLO
import cv2
import os
import numpy as np
import pyrealsense2 as rs
import time
from pathlib import Path
OUT_IMG = "dataset/images/test"
PRED_IMG = "dataset/images/predictions"
os.makedirs(PRED_IMG, exist_ok=True)

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
    def __init__(self, classes, mode="test"):
        self.W=640
        self.H=480
        self.mode = mode

        # Initialize a YOLO model (Prefer TensorRT engine if available)
        #name = "yolo26n_bee_beetle_butterfly"
        name = "yolo26n_jutestripe_yellowpaper"
        model_pt = f"/model/{name}-seg.pt"
        device_suffix = os.getenv("DEVICE_SUFFIX", "") # e.g. "_pc" or "_jetson"
        model_engine = f"/model/{name}-seg{device_suffix}.engine"

        if os.path.exists(model_engine):
            print(f"Loading TensorRT engine: {model_engine}")
            self.model = YOLO(model_engine, task="segment")
        else:
            print(f"Loading PyTorch model: {model_pt}")
            self.model = YOLO(model_pt)
        # Save classes to detect
        self.classes = classes
        self.class_ids = [id for id in self.model.names if self.model.names[id] in classes]

        # Initialize webcam
        #self.cap = cv2.VideoCapture(0)
        if self.mode != "test":
            self.initialize_realsense()

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self):
        # init realsense

        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 30)
        self.rs_config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 30)
        self.rs_pipeline.start(self.rs_config)


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
        if self.mode != "test":
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            color_image = cv2.imread(os.path.join(OUT_IMG, out_name))
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
            conf=0.5,
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
        rs.pipeline().stop()
        #self.cap.release()
        cv2.destroyAllWindows()
        pass

    def run(self):
        if self.mode == "test":
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
                cv2.imwrite(os.path.join(PRED_IMG, f"pred_{out_name}"), annotated_color_image)

                # Beenden mit 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            while True:
                # Kamerabild lesen
                color_image, depth_image = self.get_frame(None)

                # Objekte detektieren
                frame_masks, annotated_color_image = self.detect_obj(color_image)

                # RGB- und Tiefenbild fussionieren
                #coordinates = self.fuse(color_image, depth_image, frame_masks)
                # Objekt-Koordinaten berechnen

                # Detektion anzeigen
                cv2.imshow("Detektion", annotated_color_image)

                # Beenden mit 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.stop_camera()
        
        

if __name__ == "__main__":
    #person_detection = ObjDetection(["bee", "beetle", "butterfly"], mode="camera")
    person_detection = ObjDetection(["jute-stripe", "yellow-paper"], mode="test")
    person_detection.run()