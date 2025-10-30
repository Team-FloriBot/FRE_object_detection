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
        self.cap = cv2.VideoCapture(0)

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self):
        # init realsense
        pass

    # ---------------------------------------------------------
    # Capture Frame in camera
    # ---------------------------------------------------------
    def get_frame(self):
        """""""""""""""""""""""""""
        RGB- und Tiefenbild lesen
        Outputs: color_image, depth_image
        """""""""""""""""""""""""""
        
        # --> replace that with rs
        ret, color_image = self.cap.read()

        # --> replace that with rs
        color_image = cv2.resize(color_image, (self.W, self.H))
        depth_image = None
        return color_image, depth_image


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
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        while True:
            # Kamerabild lesen
            color_image, depth_image = self.get_frame()

            # Objekte detektieren
            frame_masks, annotated_color_image = self.detect_obj(color_image)

            # RGB- und Tiefenbild fussionieren
            coordinates = self.fuse(color_image, depth_image, frame_masks)
            # Objekt-Koordinaten berechnen

            # Detektion anzeigen
            cv2.imshow("Orginal", color_image)
            cv2.imshow("Detektion", annotated_color_image)

            # Beenden mit 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()

if __name__ == "__main__":
    # person_detection = ObjDetection(["person"])
    # person_detection.run()

    ## License: Apache 2.0. See LICENSE file in root directory.
    ## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

 ## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################



    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()