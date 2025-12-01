import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d


class ObjDetection:
    def __init__(self, classes,
                 use_decimation=False,
                 use_spatial=True,
                 use_temporal=True,
                 use_hole_filling=True,
                 use_mask_filter=True,
                 conf=0.5):

        self.W = 640
        self.H = 480
        self.conf = conf

        # ------------------------------
        # MASK R-CNN LADEN
        # ------------------------------
        print("Lade Mask R-CNN Gewichte ...")
        weights = torch.load("mask_rcnn_final_2.pth", map_location="cpu")

        # Anzahl Klassen aus Gewichten bestimmen
        num_classes = weights["roi_heads.box_predictor.cls_score.weight"].shape[0]
        print("Gefundene Klassen:", num_classes)

        # Modell erzeugen
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes
        )

        # Gewichte laden
        self.model.load_state_dict(weights)

        # GPU aktivieren
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("Mask R-CNN läuft auf der GPU")
        else:
            print("Mask R-CNN läuft auf der CPU")

        self.model.eval()

        # Transform für Bilder
        self.transform = T.Compose([T.ToTensor()])

        # --------------------------------
        # Klassenbezeichnungen generieren
        # PyTorch kennt keine Namen → wir generieren:
        # class_0, class_1, ...
        # --------------------------------
        self.id_to_name = {i: f"class_{i}" for i in range(num_classes)}

        self.classes = classes  # erwartete Klassennamen (string)
        # es kann sein, dass deine Klasse "Tennisball" class_1 ist
        # das matchen wir dynamisch
        self.class_ids = [
            cid for cid, cname in self.id_to_name.items()
            if cname in classes
        ]

        # RealSense pipeline
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.is_initialized = False

        # Filter Flags
        self.use_decimation = use_decimation
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_hole_filling = use_hole_filling
        self.use_mask_filter = use_mask_filter

        # Filter initialisieren
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
        """
        Input: RGB image (numpy)
        Output: obj_masks = [{"class": name, "mask": np.array}], annotated_image
        """

        # Bild in Tensor umwandeln
        img_tensor = self.transform(color_image)
        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")

        # Inferenz
        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        scores = outputs["scores"].cpu().numpy()
        boxes = outputs["boxes"].cpu().numpy()
        masks = outputs["masks"].cpu().numpy()  # (N,1,H,W)
        labels = outputs["labels"].cpu().numpy()

        annotated_image = color_image.copy()
        obj_masks = []

        # Schleife über alle erkannten Objekte
        for score, box, mask, label in zip(scores, boxes, masks, labels):

            # Mindestconfidence
            if score < self.conf:
                continue

            # Falls user Klassen angegeben hat
            if len(self.class_ids) > 0 and label not in self.class_ids:
                continue

            # Mask from (1,H,W) → (H,W)
            mask = mask[0]

            # Mask in 0/1 umwandeln
            binary_mask = (mask > 0.5).astype("uint8")

            # Für overlay
            mask_uint8 = (binary_mask * 255).astype("uint8")

            # Farbige Overlaymaske erzeugen
            color_mask = np.zeros_like(annotated_image)
            color_mask[:, :, 0] = mask_uint8  # blau

            annotated_image = cv2.addWeighted(annotated_image, 1,
                                              color_mask, 0.5, 0)

            obj_masks.append({
                "class": self.id_to_name[label],
                "mask": binary_mask
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
    test_object=ObjDetection(["Tennisball"], conf=0.7)
    test_object.run()
 