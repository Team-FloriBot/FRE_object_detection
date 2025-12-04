import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from ultralytics import YOLO
import tracking

class ObjDetection:
    def __init__(self, classes,
                 model_type="rcnn",          # Select model: 'yolo' or 'rcnn'
                 use_decimation=False,       # Decreases resolution, loss of precision at edges
                 use_spatial=True,           # Smooths edges (good for walls, bad for small floating objects)
                 use_temporal=False,         # Filters over time (can cause ghosting if objects move fast)
                 use_hole_filling=True,      # Fills missing depth data with estimated values
                 use_mask_filter=True,       # Post-processing of segmentation masks (erode/dilate)
                 use_localization_factor=True, # Apply correction factors to 3D coordinates
                 conf=0.5):

        # select frame resolution
        self.W = 640
        self.H = 480

        # --- Tracking Configuration ---
        # Initialize tracker for temporal mask filtering; reduces noise in localization
        if model_type == "yolo":
            min_hits = 20
            max_dist = 80       # Max pixels an object is allowed to move per frame
            max_missing = 3     # How many frames an object can be lost before deletion
            self.tracker = tracking.ObjectTracker(max_missing=max_missing, min_hits=min_hits, max_dist=max_dist)

        if model_type == "rcnn":
            min_hits = 5 # rcnn is much slower
            max_dist = 80       # Max pixels an object is allowed to move per frame
            max_missing = 5     # How many frames an object can be lost before deletion
            self.tracker = tracking.ObjectTracker(max_missing=max_missing, min_hits=min_hits, max_dist=max_dist)


        # --- Coordinate Correction Factors ---
        self.use_localization_factor = use_localization_factor
        self.factor_x = 0.5919
        self.factor_y = 0.5409
        self.factor_z = 0.9344

        # --- Internal States and general initializations ---
        self.conf = conf
        self.model_type = model_type.lower()
        self.classes = classes
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.is_initialized = False
        # Camera Intrinsics
        self.cx = 0
        self.cy = 0
        self.fx = 0
        self.fy = 0

        # --- Model Selection & Initialization --- 
        if self.model_type == "yolo":
            print("Initializing YOLO Model...")
            # Initialize a YOLO model
            self.model = YOLO("tennisball_600_seg_yolo11_v02.pt")
            
            # Save classes to detect (Filter logic for YOLO)
            self.class_ids = [id for id in self.model.names if self.model.names[id] in classes]
            
            if torch.cuda.is_available():
                self.model.to('cuda')
                print("YOLO runs on the GPU (CUDA enabled).")
            else:
                print("No GPU found, YOLO is running on the CPU.")

        elif self.model_type == "rcnn":
            print("Initializing Mask R-CNN Model...")
            print("Loading Mask R-CNN weights...")
            # Load specific weights file
            weights = torch.load("mask_rcnn_final_3.pth", map_location="cpu")

            # Determine number of classes from weights
            num_classes = weights["roi_heads.box_predictor.cls_score.weight"].shape[0]
            print("Classes found in weights:", num_classes)

            # Create Model
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=None,
                num_classes=num_classes
            )

            # Load state dictionary
            self.model.load_state_dict(weights)

            if torch.cuda.is_available():
                self.model.to("cuda")
                print("Mask R-CNN runs on the GPU")
            else:
                print("Mask R-CNN runs on the CPU")

            self.model.eval()

            # Transform for images
            self.transform = T.Compose([T.ToTensor()])

            # Generate generic class names (class_0, class_1...) because PyTorch models don't store names
            self.id_to_name = {i: f"class_{i}" for i in range(num_classes)}
            
            # Filter class IDs based on user input
            self.class_ids = [
                cid for cid, cname in self.id_to_name.items()
                if cname in classes
            ]
        else:
            raise ValueError("Invalid model_type. Please choose 'yolo' or 'rcnn'.")

        # --- Filter setup ---
        self.use_decimation = use_decimation
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_hole_filling = use_hole_filling
        self.use_mask_filter = use_mask_filter

        # Decimation Filter: Reduces resolution
        self.dec_filter = rs.decimation_filter()
        self.dec_filter.set_option(rs.option.filter_magnitude, 2)

        # Temporal Filter: Uses previous frames to smooth data
        self.temp_filter = rs.temporal_filter()

        # Hole Filling: Fills invalid depth pixels
        self.hole_filter = rs.hole_filling_filter()
        self.hole_filter.set_option(rs.option.holes_fill, 2) # Mode 2: Nearest from around

        # Spatial Filter: Edge-preserving smoothing
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)      # Iterations (Medium smoothing)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5) # Smoothing strength
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Threshold (High delta preserves edges)
        self.spatial_filter.set_option(rs.option.holes_fill, 0)   
        self.spatial_filter.set_option(rs.option.holes_fill, 0) # Do not fill holes via spatial filter, hole filling fiter does this

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self, color_resolution=(640, 480), fps=30):
        """
        Initialises the RealSense pipeline and starts streaming.
        """
        depth_resolution = (self.W, self.H)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, fps)

        profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.is_initialized = True
        print("RealSense initialized successfully.")

    # ---------------------------------------------------------
    # Capture Frame
    # ---------------------------------------------------------
    def get_frame(self):
        """
        Read and wait for frames.
        """
        if not self.is_initialized:
            raise RuntimeError("RealSense pipeline not initialized. Call initialize_realsense() first.")

        frames = self.pipeline.wait_for_frames()
        return frames

    # ---------------------------------------------------------
    # Align Frames & Apply Filters
    # ---------------------------------------------------------
    def align_frames(self, frames):
        if not self.align:
            raise RuntimeError("Alignment object not initialized.")

        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # Apply depth filters
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
    # Detection Wrapper
    # ---------------------------------------------------------
    def detect_obj(self, color_image):
        """
        Routes the detection to the selected model type.
        """
        if self.model_type == "yolo":
            return self._detect_yolo(color_image)
        elif self.model_type == "rcnn":
            return self._detect_rcnn(color_image)

    # ---------------------------------------------------------
    # YOLO Implementation
    # ---------------------------------------------------------
    def _detect_yolo(self, color_image):
        results = self.model.predict(color_image, classes=self.class_ids, conf=self.conf, imgsz=color_image.shape[:2], verbose=False)
        annotated_image = color_image.copy()
        raw_detections = []

        result = results[0]

        if result.masks is not None:
            mask_counter = 0
            for box, mask_tensor in zip(result.boxes, result.masks.data):
                # Convert mask tensor to numpy
                mask = mask_tensor.cpu().numpy()

                # Resize if necessary
                if mask.shape != annotated_image.shape[:2]:
                    mask = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                # prepare object tracker
                det_entry = {
                    "class": self.model.names[int(box.cls[0])],
                    "mask": mask,
                    "center": self._get_center(mask) # Mittelpunkt berechnen
                }
                raw_detections.append(det_entry)
        
        confirmed_objects = self.tracker.update(raw_detections)

        # Visualization only for confirmed objects                
        for obj in confirmed_objects:
            mask = obj["mask"]

            # Visualization (Blue Overlay)
            mask_uint8 = (mask * 255).astype("uint8")
            color_mask = np.zeros_like(annotated_image)
            color_mask[:, :, 0] = mask_uint8

            annotated_image = cv2.addWeighted(annotated_image, 1, color_mask, 0.5, 0)

        return confirmed_objects, annotated_image

    # ---------------------------------------------------------
    # R-CNN Implementation
    # ---------------------------------------------------------
    def _detect_rcnn(self, color_image):
        img_tensor = self.transform(color_image)
        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        scores = outputs["scores"].cpu().numpy()
        masks = outputs["masks"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        annotated_image = color_image.copy()
        raw_detections = []

        for score, mask, label in zip(scores, masks, labels):
            if score < self.conf: continue
            if len(self.class_ids) > 0 and label not in self.class_ids: continue

            mask = mask[0]
            binary_mask = (mask > 0.5).astype("uint8")
            
            det_entry = {
                "class": self.id_to_name[label],
                "mask": binary_mask,
                "center": self._get_center(binary_mask)
            }
            raw_detections.append(det_entry)

        # --- TRACKER UPDATE ---
        confirmed_objects = self.tracker.update(raw_detections)

        # --- VISUALISIERUNG ---
        for obj in confirmed_objects:
            mask_uint8 = (obj["mask"] * 255).astype("uint8")
            color_mask = np.zeros_like(annotated_image)
            color_mask[:, :, 0] = mask_uint8
            annotated_image = cv2.addWeighted(annotated_image, 1, color_mask, 0.5, 0)

        return confirmed_objects, annotated_image

    # ---------------------------------------------------------
    # Fusion (Common for both models)
    # ---------------------------------------------------------
    def fuse(self, color_image, depth_image, obj_masks):
        """
        Generates point clouds from color and depth images for each detected object.
        """
        # Camera parameters of the depth camera
        intrinsics_depth = self.pipeline.get_active_profile().get_stream(rs.stream.depth)\
                                        .as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy 

        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

        point_cloud_results = {}
        depth_masked = np.zeros_like(depth_image, dtype=depth_image.dtype)
        instance_counter = 0

        for obj in obj_masks:
            label = obj["class"]
            
            # Ensure mask is binary (0 or 1)
            mask = (obj["mask"] > 0.5).astype(np.uint8) 

            # Mask filtering: Close
            if self.use_mask_filter:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

                # Mask filtering: Erode (cut away unsafe edges)
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.erode(mask, kernel_erode, iterations=2)

            # Resize mask to match depth image if needed
            if mask.shape != depth_image.shape:
                mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            
            # Resize color image to match depth image if needed
            if color_image.shape[:2] != depth_image.shape:
                color_image_proc = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                color_image_proc = color_image

            # Pixel coordinates of the mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            # Fill masked depth image (for visualization)
            depth_masked[ys, xs] = depth_image[ys, xs]

            # Depth values in meters
            z = depth_image[ys, xs].astype(float) * self.depth_scale # depth parallel to optical axis
            
            # Depth filter: valid values
            valid = (z > 0.1) & (z < 20.0) & (~np.isnan(z))
            xs, ys, z = xs[valid], ys[valid], z[valid]
            if len(xs) == 0:
                continue

            # 3D coordinates (Camera Coordinate System)
            if self.use_localization_factor:
                X = (xs - cx) * z / fx * self.factor_x
                Y = (ys - cy) * z / fy  * self.factor_y
                Z = z * self.factor_z

            else:
                X = (xs - cx) * z / fx
                Y = (ys - cy) * z / fy
                Z = z

            # Invert Y for Open3D
            Y = -Y

            points = np.stack((X, Y, Z), axis=-1)

            # Get color values (BGR -> RGB)
            colors = color_image_proc[ys, xs][:, ::-1] / 255.0

             # Calculate median
            median_xyz = np.median(points, axis=0)

            # Z-Filtering: Allow only points within tolerance of median Z
            z_median = median_xyz[2]
            z_values = points[:, 2]
            mask_z = np.abs(z_values - z_median) < 0.05 
            
            points = points[mask_z]
            colors = colors[mask_z]       
            
            if len(points) == 0:
                continue
            
            median_xyz = np.median(points, axis=0) # recalculate median    

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
        if self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")

    # --- HELP FUNCTION: Calculate the centre point of the mask ---
    def _get_center(self, mask):
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Fallback if mask is empty or only noise
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cX, cY = int(np.mean(xs)), int(np.mean(ys))
            else:
                cX, cY = 0, 0
        return (cX, cY)

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------
    def run(self):
        """
        Main loop: Read, merge, and display frames.
        Exit with 'q'.
        """
        if not self.is_initialized:
            self.initialize_realsense()

        print(f"Starting camera stream using {self.model_type.upper()}... Press 'q' to quit.")

        # Start Open3D Visualizer once
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live 3D Point Cloud", width=900, height=700)
        pcd = o3d.geometry.PointCloud()
   
        vis.add_geometry(pcd)   # Point cloud
        vis.add_geometry(axis)  # Add axes

        try:
            while True:
                # Read frames
                frames = self.get_frame()

                # Align frames
                color_image, depth_image = self.align_frames(frames)
                if color_image is None or depth_image is None:
                    continue
                
                # Detect Objects (Using the selected model)
                obj_masks, annotated_color_image = self.detect_obj(color_image)

                # Fuse RGB + Depth to point clouds
                pc_dict, depth_image_masked = self.fuse(color_image, depth_image, obj_masks)

                # Masked depth image for visualization
                depth_masked_normalized = cv2.normalize(depth_image_masked, None, 0, 255, cv2.NORM_MINMAX)
                depth_masked_normalized = depth_masked_normalized.astype(np.uint8)
                depth_image_masked_color = cv2.applyColorMap(depth_masked_normalized, cv2.COLORMAP_JET)

                # Write median coordinates on the image
                for instance in pc_dict.values():
                    median = instance["median"]
                    # Approximate pixel coordinates of the median
                    if median[2] != 0:
                        x_pixel = int((median[0] * self.fx) / median[2] + self.cx)
                        y_pixel = int((-median[1] * self.fy) / median[2] + self.cy) # Invert Y again for image coordinates
                        text = f"X:{median[0]:.2f} Y:{median[1]:.2f} Z:{median[2]:.2f}"
                        cv2.putText(annotated_color_image, text, (x_pixel, y_pixel),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Show results (OpenCV)
                cv2.imshow("Original", color_image)
                cv2.imshow("Detection - RGB", annotated_color_image)
                if depth_image_masked is not None:
                    cv2.imshow("Detection - Depth - Mask", depth_image_masked_color)

                # Update Open3D Point Cloud live
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
                else:
                    # Clear point cloud if no object detected (optional)
                    # pcd.points = o3d.utility.Vector3dVector([])
                    # vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()

                # Quit with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            vis.destroy_window()
            self.stop_camera()

if __name__ == "__main__":
    # EXAMPLE USE:
    # Set model_type="yolo" OR model_type="rcnn"
    
    # 1. Using YOLO
    # test_object = ObjDetection(["Tennisball"], model_type="yolo", conf=0.5)
    
    # 2. Using R-CNN (Ensure class name matches "class_X" or similar logic if using generic weights)
    test_object = ObjDetection(["Tennisball"])
    
    test_object.run()