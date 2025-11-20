from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import open3d as o3d
import torch


class ObjDetection:
    def __init__(self, classes,
                 use_decimation=False,
                 use_spatial=True,
                 use_temporal=True,
                 use_hole_filling=True,
                 use_mask_filter=True):
        self.W=640
        self.H=480

        # Initialize a YOLO model
        self.model = YOLO("yolov8n-seg.pt")

        # >>> GPU aktivieren, falls verfügbar <<<
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("YOLOv8 läuft auf der GPU (CUDA aktiviert).")
        else:
            print("Keine GPU gefunden, YOLO läuft auf der CPU.")

        # Save classes to detect
        self.classes = classes
        self.class_ids = [id for id in self.model.names if self.model.names[id] in classes]

        # Parameter to initialize RealSense 
        self.pipeline = None
        self.config = None

        # Parameters to Fuse Images
        self.align = None
        self.depth_scale = None
        self.is_initialized = False

        # Set filters on/off
        self.use_decimation = use_decimation
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_hole_filling = use_hole_filling
        self.use_mask_filter = use_mask_filter

        # initialize depth filters
        self.dec_filter = rs.decimation_filter()      
        self.spatial_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()        
        self.hole_filter = rs.hole_filling_filter()   

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
        read and align RGB- and Depth-Frame
        Outputs: color_image, depth_image
        """""""""""""""""""""""""""
        
        if not self.is_initialized:
            raise RuntimeError("RealSense pipeline not initialized. Call Initialize_RealSense() first.")

        frames = self.pipeline.wait_for_frames()

        return frames
        

    # ---------------------------------------------------------
    # Align captured frames
    # ---------------------------------------------------------
    def align_frames(self, frames):
        if not self.align:
            raise RuntimeError("Alignment object not initialized. Run Initialize_RealSense() first.")

        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # apply depth filters 
        if aligned_depth_frame:
            if self.use_decimation:
                aligned_depth_frame = self.dec_filter.process(aligned_depth_frame)
            if self.use_spatial:
                aligned_depth_frame = self.spatial_filter.process(aligned_depth_frame)
            if self.use_temporal:
                aligned_depth_frame = self.temp_filter.process(aligned_depth_frame)
            if self.use_hole_filling:
                aligned_depth_frame = self.hole_filter.process(aligned_depth_frame)

        if not aligned_depth_frame or not aligned_color_frame:
            return None, None

        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        return aligned_color_image, aligned_depth_image

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

        obj_masks = []

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
                obj_masks.append({
                    "class": self.model.names[int(box.cls[0])],
                    "mask": mask
                })


        return obj_masks, annotated_image

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------
    def fuse(self, color_image, depth_image, obj_masks):
        """
        Erzeugt Punktwolken aus Farb- und Tiefenbild für jedes erkannte Objekt.
        Gibt ein Dictionary zurück:
            { label_index: {
                "label": str,
                "points": Nx3,
                "colors": Nx3,
                "median": array([X_m, Y_m, Z_m])
            } }
        """

        # Kameraparameter der Tiefenkamera (nicht Farbkamera!)
        intrinsics_depth = self.pipeline.get_active_profile().get_stream(rs.stream.depth)\
                       .as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy 

        self.cx=cx
        self.cy=cy
        self.fx=fx
        self.fy=fy

        point_cloud_results = {}

        # Maske für das Tiefenbild initialisieren
        depth_masked = np.zeros_like(depth_image, dtype=depth_image.dtype)

        instance_counter = 0  # eindeutige ID für jede Instanz

        for obj in obj_masks:
            label = obj["class"]

            # Maske binär
            mask = (obj["mask"] > 0.5).astype(np.uint8) 

            # Maskenfilterung: Morphologie
            if self.use_mask_filter:
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Maske an Tiefenauflösung anpassen
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

            # Pixelkoordinaten der Maske
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            # Maskiertes Tiefenbild füllen
            depth_masked[ys, xs] = depth_image[ys, xs]

            # Tiefenwerte in Meter
            z = depth_image[ys, xs].astype(float) * self.depth_scale
            
            # Tiefenfilter: gültige Werte
            valid = (z > 0.1) & (z < 10.0) & (~np.isnan(z))
            xs, ys, z = xs[valid], ys[valid], z[valid]
            if len(xs) == 0:
                continue

            # 3D-Koordinaten (Kamera-Koordinatensystem)
            X = (xs - cx) * z / fx
            Y = (ys - cy) * z / fy
            Z = z

            # invertierung für Open 3d
            Y=-Y

            points = np.stack((X, Y, Z), axis=-1)

            # Farbwerte an denselben Pixeln holen (BGR → RGB)
            colors = color_image[ys, xs][:, ::-1] / 255.0

             # Median berechnen
            median_xyz = np.median(points, axis=0)         

            point_cloud_results[instance_counter] = {
                "label": label,
                "points": points,
                "colors": colors,
                "median": median_xyz
            }
            instance_counter += 1


        return point_cloud_results, depth_masked
    
    
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

        # Open3D Visualizer einmalig starten
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live 3D Point Cloud", width=900, height=700)
        pcd = o3d.geometry.PointCloud()
   
        vis.add_geometry(pcd)   # deine Punktwolke
        vis.add_geometry(axis)  # Achsen hinzufügen

        geom_added = True

        try:
            while True:
            
                # read cameraframes
                frames = self.get_frame()

                #Align frames
                color_image, depth_image = self.align_frames(frames)
                if color_image is None or depth_image is None:
                    continue
                
                # Detect Objects
                obj_masks, annotated_color_image = self.detect_obj(color_image)

                # Fuse RGB + Depth zu Punktwolken
                pc_dict, depth_image_masked = self.fuse(color_image, depth_image, obj_masks)

                # Makiertes Tiefenbild
                depth_masked_normalized = cv2.normalize(depth_image_masked, None, 0,255,cv2.NORM_MINMAX)
                depth_masked_normalized= depth_masked_normalized.astype(np.uint8)
                depth_image_masked_color =cv2.applyColorMap(depth_masked_normalized, cv2.COLORMAP_JET)

                # Mediankoordinaten auf das Bild schreiben
                for instance in pc_dict.values():
                    median = instance["median"]
                    # Pixelkoordinaten des Medians approximieren
                    x_pixel = int((median[0] * self.fx) / median[2] + self.cx)
                    y_pixel = int((-median[1] * self.fy) / median[2] + self.cy)  # invert Y wieder für Bildkoordinaten
                    text = f"X:{median[0]:.2f} Y:{median[1]:.2f} Z:{median[2]:.2f}"
                    cv2.putText(annotated_color_image, text, (x_pixel, y_pixel),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Show results
                cv2.imshow("Orginal", color_image)
                cv2.imshow("Detektion - RGB", annotated_color_image)
                if depth_image_masked is not None:
                    cv2.imshow("Detektion - Depth - Mask", depth_image_masked_color)

                # Open3D Punktwolke live aktualisieren
                if len(pc_dict) > 0:
                    all_points = []
                    all_colors = []

                    for data in pc_dict.values():
                        all_points.append(data["points"])
                        all_colors.append(data["colors"])

                    points = np.concatenate(all_points, axis=0)
                    colors = np.concatenate(all_colors, axis=0)

                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis.update_geometry(pcd)

                    vis.poll_events()
                    vis.update_renderer()

                # Beenden mit 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            vis.destroy_window()
            self.stop_camera()

if __name__ == "__main__":
    test_object=ObjDetection(["person"])
    test_object.run()
 