import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from ultralytics import YOLO
import tracking
from sklearn.neighbors import NearestNeighbors
class ObjDetection:
    def __init__(self, classes,
                 model_type="yolo",          # Select model: 'yolo' or 'rcnn'
                 use_decimation=False,       # Decreases resolution, loss of precision at edges
                 use_spatial=False,           # Smooths edges (good for walls, bad for small floating objects)
                 use_temporal=True,         # Filters over time (can cause ghosting if objects move fast)
                 use_hole_filling=True,      # Fills missing depth data with estimated values
                 use_mask_filter=True,       # Post-processing of segmentation masks (erode/dilate)
                 use_localization_factor=True, # Apply correction factors to 3D coordinates
                 conf=0.5):

        # select frame resolutionqqqqqqqq
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
        self.factor_x = 1
        self.factor_y = 0.9
        self.factor_z = 0.9

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
        temp_filter = rs.temporal_filter()
        temp_filter.set_option(rs.option.filter_smooth_alpha, 0.15) # Stärkere Glättung
        temp_filter.set_option(rs.option.filter_smooth_delta, 40)   # Rausch-Toleranz erhöht
        temp_filter.set_option(rs.option.holes_fill, 0) # Do not fill holes via temporal filter, hole filling fiter does this
        self.temp_filter = temp_filter
        
        # Hole Filling: Fills invalid depth pixels
        self.hole_filter = rs.hole_filling_filter()
        self.hole_filter.set_option(rs.option.holes_fill, 1) # Mode 1: Farthest from around

        # Spatial Filter: Edge-preserving smoothing
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)      # Iterations (Medium smoothing)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5) # Smoothing strength
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # Threshold (High delta preserves edges)
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

        # Camera parameters of the depth frame for localization
        intrinsics_depth = aligned_frames.get_depth_frame().get_profile().as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy 

        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy


        return aligned_frames
    
    def depth_filter(self, aligned_frames):

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            return None, None

        # Apply depth filters
        filtered_depth_frame = aligned_depth_frame
        if self.use_decimation:
            filtered_depth_frame = self.dec_filter.process(filtered_depth_frame)
        if self.use_spatial:
            filtered_depth_frame = self.spatial_filter.process(filtered_depth_frame)
        if self.use_temporal:
            filtered_depth_frame = self.temp_filter.process(filtered_depth_frame)
        if self.use_hole_filling:
            filtered_depth_frame = self.hole_filter.process(filtered_depth_frame)

        filtered_depth_image = np.asanyarray(filtered_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        return aligned_color_image, filtered_depth_image

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

                # prepare object
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
    # Fusion 
    # ---------------------------------------------------------
    def fuse(self, color_image, depth_image, obj_masks):
        """
        Generates point clouds from color and depth images for each detected object.
        """

        point_cloud_results = {}
        depth_masked = np.zeros_like(depth_image, dtype=depth_image.dtype)
        
        # Prepare grid for vectorisation (once or per ROI)
        # We do this per ROI, as it is faster than doing it for the entire image.
        instance_counter = 0

        for obj in obj_masks:
            label = obj["class"]
            mask_full = (obj["mask"] > 0.5).astype(np.uint8)

            # 1. Calculate bounding box (efficiency boost: process ROI only)
            ys, xs = np.where(mask_full > 0)
            if len(xs) == 0: continue

            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)

            # Cut out ROI
            mask_roi = mask_full[y_min:y_max+1, x_min:x_max+1]
            depth_roi_raw = depth_image[y_min:y_max+1, x_min:x_max+1]
            color_roi = color_image[y_min:y_max+1, x_min:x_max+1]

            # Mask filtering (only on ROI)
            if self.use_mask_filter:
                # Close
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_close)

                # Erode (clean edges, and round edges)
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                mask_roi = cv2.erode(mask_roi, kernel_erode, iterations=2)

                # Dilatation (clean edges, and round edges)
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                mask_roi = cv2.dilate(mask_roi, kernel_dilate, iterations=2)


            # Apply mask to depth (ignore values outside the mask)
            # We copy the depth ROI so as not to change the original
            z_roi = depth_roi_raw.astype(np.float32) * self.depth_scale

            # We set pixels that do NOT belong to the mask to 0 (they will be ignored later).
            z_roi[mask_roi < 0.1] = 0.0
            z_roi[mask_roi == 0] = 0.0

            # 2. Statistical analysis & outlier detection
            # Only consider valid values within the mask
            valid_z =z_roi[z_roi > 0.1]

            if len(valid_z) == 0: continue

            median_z = np.median(valid_z)

            # Define tolerance range (e.g. +/- 15 cm around median)
            z_threshold = 0.1

            # Bad Pixel Mask: Pixels IN THE MASK, but Z is 0 or deviates significantly
            # We want to interpolate these instead of deleting them!
            bad_pixel_mask = np.zeros_like(mask_roi, dtype=np.uint8)

            # Criteria for "bad pixels" within the object mask:
            # 1. Invalid depth (0 or NaN)
            # 2. Too far from the median (noise)
            is_noise = np.abs(z_roi - median_z) > z_threshold
            is_invalid = (z_roi <= 0.001) | np.isnan(z_roi)
            
            bad_pixel_mask[(mask_roi > 0) & (is_noise | is_invalid)] = 1  # Markiere Fehler
            
            # 3. INTERPOLATION (Inpainting)
            # cv2.inpaint expects 8-bit. We temporarily scale the relevant Z-range.
            if np.sum(bad_pixel_mask) > 0:
                # Wir definieren min/max basierend auf dem Median, um Kontrast zu maximieren
                local_min = max(0, median_z - z_threshold)
                local_max = median_z + z_threshold
                
                # Clip & Normalize auf 0-255
                z_roi_clipped = np.clip(z_roi, local_min, local_max)
                z_norm = ((z_roi_clipped - local_min) / (local_max - local_min) * 255).astype(np.uint8)
                
                telea_mask = bad_pixel_mask.copy()
                telea_mask[mask_roi == 0] = 1  # Der gesamte Hintergrund wird als "Loch" markiert


                # Inpainting: Uses healthy neighbouring pixels to fill holes (Telea algorithm)
                z_inpainted_8u = cv2.inpaint(z_norm, telea_mask, 3, cv2.INPAINT_TELEA)
                
                # Scaling back to metres
                z_inpainted = (z_inpainted_8u.astype(np.float32) / 255.0) * (local_max - local_min) + local_min
                
                # Only accept the repaired areas
                z_roi[bad_pixel_mask > 0] = z_inpainted[bad_pixel_mask > 0]

            # 4. Projection in 3D (vectorised calculation)
            # Create coordinate grid for the ROI
            # Grid coordinates relative to the overall image
            grid_y, grid_x = np.meshgrid(
                np.arange(y_min, y_max + 1), 
                np.arange(x_min, x_max + 1), 
                indexing='ij'
            )

            # Only process masked pixels (now including interpolated pixels!)
            valid_mask = (mask_roi > 0)

            if not np.any(valid_mask): continue

            # Extract vectors
            z_final = z_roi[valid_mask]
            x_final = grid_x[valid_mask]
            y_final = grid_y[valid_mask]
            colors_final = color_roi[valid_mask][:, ::-1] / 255.0 # BGR -> RGB

            # 3D calculation
            if self.use_localization_factor:
                X = (x_final - self.cx) * z_final / self.fx * self.factor_x
                Y = (y_final - 5 - self.cy) * z_final / self.fy * self.factor_y
                Z = z_final * self.factor_z
            else:
                X = (x_final - self.cx) * z_final / self.fx
                Y = (y_final - self.cy) * z_final / self.fy
                Z = z_final

            Y = -Y # Open3D convention

            points = np.stack((X, Y, Z), axis=-1)

            nbrs = NearestNeighbors(n_neighbors=10).fit(points)
            distances, _ = nbrs.kneighbors(points)

            mean_dist = distances[:,1:].mean(axis=1)
            mad = np.median(np.abs(mean_dist - np.median(mean_dist)))
            threshold = np.median(mean_dist) + 3 * mad

            mean_dist < threshold

            keep_knn = mean_dist < threshold

            points = points[keep_knn]
            colors_final = colors_final[keep_knn]

            # Visualisation: Update masked depth image (for display only)
            # We pack the (possibly interpolated) raw values back
            depth_roi_fixed_raw = (z_roi / self.depth_scale).astype(np.uint16)
            depth_masked[y_min:y_max+1, x_min:x_max+1] = np.where(mask_roi > 0, depth_roi_fixed_raw, depth_masked[y_min:y_max+1, x_min:x_max+1])

            # Median für Label berechnen
            median_xyz = np.mean(points, axis=0)

             # Calculate median
            #median_xyz = np.median(points, axis=0)

            # Z-Filtering: Allow only points within tolerance of median Z
            z_median = median_xyz[2]
            z_values = points[:, 2]
            mask_z = np.abs(z_values - z_median) < 0.05
            
            points = points[mask_z]
            colors_final = colors_final[mask_z]       
            
            if len(points) == 0:
                continue


            # Kugelfit
            center, radius = self.fit_sphere(points)
            d = np.linalg.norm(points - center, axis=1)

            # Radiale Qualität
            radial_quality = 1 - np.std(d) / radius
            radial_quality = np.clip(radial_quality, 0, 1)

            # 3D-Verteilung
            X = points - points.mean(axis=0)
            cov = np.cov(X.T)
            l = np.linalg.eigvalsh(cov)
            spatial_quality = l.min() / l.max()
            #spatial_quality = l[0] / np.mean(l)

            sphere_quality = min(10 * 100 * radial_quality * spatial_quality,100)
            

            median_xyz = np.median(points, axis=0) # recalculate median   


            point_cloud_results[instance_counter] = {
                "label": label,
                "points": points,
                "colors": colors_final,
                "median": median_xyz,
                "balliness": sphere_quality,
                "radius": radius

            }
            instance_counter += 1

        return point_cloud_results, depth_masked

    def fit_sphere(self, points):
        # points: Nx3 numpy array
        
        # Extract x,y,z
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        # Build the A matrix and f vector for least squares
        A = np.column_stack([2*x, 2*y, 2*z, np.ones_like(x)])
        f = x**2 + y**2 + z**2

        # Solve A*p = f for p = [x0, y0, z0, c]
        C, *_ = np.linalg.lstsq(A, f, rcond=None)

        x0, y0, z0, c = C
        # compute radius
        r = np.sqrt(c + x0**2 + y0**2 + z0**2)

        return np.array([x0, y0, z0]), r
    
    def sphere_fit_error(self, points, center, radius):
        d = np.linalg.norm(points - center, axis=1)
        residuals = d - radius
        rmse = np.sqrt(np.mean(residuals**2))
        return rmse, residuals
    
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
                aligned_frames = self.align_frames(frames)
                color_image, depth_image = self.depth_filter(aligned_frames)

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
                    balliness = instance["balliness"]
                    radius = instance["radius"]
                    # Approximate pixel coordinates of the median
                    if median[2] != 0:
                        x_pixel = int((median[0] * self.fx) / median[2] + self.cx)
                        y_pixel = int((-median[1] * self.fy) / median[2] + self.cy) # Invert Y again for image coordinates
                        text = f"X:{median[0]:.2f} Y:{median[1]:.2f} Z:{median[2]:.2f} B:{balliness:.0f}% R:{radius:.2f}"
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