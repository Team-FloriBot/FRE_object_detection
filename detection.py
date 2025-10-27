from ultralytics import YOLO
import cv2
import pyrealsense2 as rs

class Detection:
    def __init__(self, model_name):
        self.models = []
        self.model_paths = []

        # Model(s) laden
        if isinstance(model_name, str):
            model_path = f"../models/{model_name}.pt"
            self.models.append(YOLO(model_path))

        elif isinstance(model_name, (list, tuple)) and all(isinstance(x, str) for x in model_name):
            for name in model_name:
                model_paths = f"../models/{name}.pt"
                self.models.append(YOLO(model_paths))

        else:
            raise ValueError("model_name must be a string or list of strings.")


    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self):
        # init realsense
        pass

    # ---------------------------------------------------------
    # Capture Frame von der Kamera
    # ---------------------------------------------------------
    def get_frame(self):
        # RGB- und Tiefenbild lesen
        # Outputs: color_image, depth_image
        pass

    # ---------------------------------------------------------
    # Run Detection
    # ---------------------------------------------------------
    def detect(self, color_image):
        # detect objects in frame
        # Input color_image
        # Output yolo_results --> classes, position 
        pass

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------
    def fusion(self, color_image, depth_image, yolo_results):
        # calculate coordinates of detected images
        # Input color_image, depth_image, yolo_results
        # Output coordinates_results --> {label1: [[x1_1,y1_1,z1_1], [x1_2,y1_2,z1_2]], label2: [[x2_1,y2_1,z2_1]]}
        pass

    # ---------------------------------------------------------
    # Stop Kamera
    # ---------------------------------------------------------
    def stop(self):
        pass
