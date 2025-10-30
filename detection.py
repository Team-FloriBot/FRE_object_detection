from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures



class ObjDetection:
    def __init__(self, classes):
        self.W=640
        self.H=480

        # Initialize a YOLOE model
        self.model = YOLO("yolov8n-seg.pt")
        # Save classes to detect
        self.classes = classes
        self.class_ids = [id for id in self.model.names if self.model.names[id] in classes]

        # Initialize webcam
        # self.cap = cv2.VideoCapture(0)

        # Parameter to initialize RealSense 
        self.pipeline = None
        self.config = None

        # Parameters to Fuse Images
        self.align = None
        self.depth_scale = None
        #self.clipping_distance = None
        self.is_initialized = False       

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self, depth_resolution=(640, 480), color_resolution=(640, 480), fps=30):
        """
        Initialisiert RealSense-Pipeline und startet Streaming.
        """
        # Create pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, fps)

        # Start streaming
        profile = self.pipeline.start(self.config)

        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")

        # Create alignment object (align depth to color)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.is_initialized = True
        print("RealSense initialized successfully.")

    # ---------------------------------------------------------
    # Capture Frame in camera
    # ---------------------------------------------------------
    def get_frame(self):
        """""""""""""""""""""""""""
        RGB- und Tiefenbild lesen
        Outputs: color_image, depth_image
        """""""""""""""""""""""""""
        
        if not self.is_initialized:
            raise RuntimeError("RealSense pipeline not initialized. Call Initialize_RealSense() first.")

        frames = self.pipeline.wait_for_frames()
        return frames


    # ---------------------------------------------------------
    # Run Detection
    # ---------------------------------------------------------
    def detect_obj(self, color_image):
        """""""""""""""""""""""""""
        detect objects in frame
        Input color_image
        Output annotated_image, frame_mask --> classes, mask
        """""""""""""""""""""""""""

        results = self.model.predict(color_image, classes=self.class_ids)

        annotated_image = color_image.copy()

        frame_masks = []

        # Get the result object (YOLO returns a list, one item per image)
        result = results[0]

        if result.masks is not None:
            # Each entry in result.masks.data corresponds to a detected object's mask
            for box, mask_tensor in zip(result.boxes, result.masks.data):
                # Convert the mask tensor to a NumPy array
                mask = mask_tensor.cpu().numpy()

                # Convert mask values from [0, 1] to [0, 255] for visualization
                mask_uint8 = (mask * 255).astype("uint8")

                # Create a blue overlay (BGR color order)
                color_mask = np.zeros_like(annotated_image)
                color_mask[:, :, 0] = mask_uint8  # Fill blue channel

                # Blend the mask overlay with the original image
                alpha = 0.5  # Transparency factor (0 = transparent, 1 = opaque)
                annotated_image = cv2.addWeighted(annotated_image, 1, color_mask, alpha, 0)

                # Store the mask and its corresponding class name
                frame_masks.append({
                    "class": self.model.names[int(box.cls[0])],
                    "mask": mask
                })


        return frame_masks, annotated_image

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------
    def fuse(self, frames, frame_masks):
        """""""""""""""""""""""""""
        calculate coordinates of detected images
        Input color_image, depth_image, frame_masks
        Output coordinates_results --> {label1: [[x1_1,y1_1,z1_1], [x1_2,y1_2,z1_2]], label2: [[x2_1,y2_1,z2_1]]}
        """""""""""""""""""""""""""
        if not self.align:
            raise RuntimeError("Alignment object not initialized. Run Initialize_RealSense() first.")

        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            return None, None

        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        return aligned_color_image, aligned_depth_image

    # ---------------------------------------------------------
    # Stop Camera
    # ---------------------------------------------------------
    def stop_camera(self):
        """
        Stoppt die RealSense-Pipeline.
        """
        if self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")

    def run(self):
        """
        Hauptschleife: Frames lesen, fusionieren, anzeigen.
        Beenden mit 'q'.
        """
        if not self.is_initialized:
            self.initialize_realsense()

        print("Starting camera stream... Press 'q' to quit.")

        try:
            while True:
            
                # Kamerabild lesen
                #color_image, depth_image = self.get_frame()
                frames = self.get_frame()

                # Objekte detektieren
                #frame_masks, annotated_color_image = self.detect_obj(color_image)

                # Farbbild und Tiefenbild (unaligned)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # RGB- und Tiefenbild fussionieren
                # Alignment durchführen
                aligned_color_image, aligned_depth_image = self.fuse(frames)
                if aligned_color_image is None:
                    continue
                #coordinates = self.fuse(color_image, depth_image, frame_masks)
                # Objekt-Koordinaten berechnen

                # Detektion anzeigen
                cv2.imshow("Orginal", color_image)
                #cv2.imshow("Detektion", annotated_color_image)

                # Beenden mit 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.stop_camera()

if __name__ == "__main__":
    # person_detection = ObjDetection(["person"])
    # person_detection.run()
    test_object=ObjDetection(["test"])
    test_object.run()
 