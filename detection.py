from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import torch


class ObjDetection:
    def __init__(self, classes,
                 use_decimation=False, # verringert Auflösung, Präzisionsverlust an den Kanten, Maske und Tiefe passen nicht mehr pixelgenau übereinander
                 use_spatial=True, # glättet Kanten, gut für Wände, aber schlecht für frei schwebende Objekte, an Kanten berechnet er einen Mittelwert, kann aber angepasst werden, dass er Kanten nicht verwischt
                 use_temporal=False, # Ungenauigkeit, wenn sich der Ball bewegt
                 use_hole_filling=True, # Füllt fehlende Daten am Rand mit geschätzten Werten
                 use_mask_filter=True,
                 conf=0.5):

        self.conf =conf

        # Initialize a YOLO model
        self.model = YOLO("tennisball_600_seg_yolo11_v02.pt")

        # >>> Enable GPU, if available <<<
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("YOLOv8 runs on the GPU (CUDA enabled).")
        else:
            print("No GPU found, YOLO is running on the CPU.")

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
        self.dec_magnitude = 2
        self.dec_filter.set_option(rs.option.filter_magnitude, self.dec_magnitude)  # decimation of 2    
        self.temp_filter = rs.temporal_filter()        
        self.hole_filter = rs.hole_filling_filter()   
        self.hole_filter.set_option(rs.option.holes_fill, 2) # Change mode to "Nearest" (prevents values from being smudged), 0 = fill_from_left, 1 = farest_from_around, 2 = nearest_from_around
        # set spatial filter so that it does not smooth edges
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20) # Wichtig: Hohes Delta verhindert Glätten über Kanten hinweg
        self.spatial_filter.set_option(rs.option.holes_fill, 0) # Löcher nicht durch den Spatial Filter füllen lassen

    # ---------------------------------------------------------
    # RealSense Setup
    # ---------------------------------------------------------
    def initialize_realsense(self, depth_resolution=(640, 480), color_resolution=(640, 480), fps=30):
        """
        Initialises the RealSense pipeline and starts streaming.
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
        #print(color_image.shape)
        results = self.model.predict(color_image, classes=self.class_ids, conf=self.conf, imgsz=color_image.shape[:2])
        
        annotated_image = color_image.copy()

        obj_masks = []

        # Get the result object (YOLO returns a list, one item per image)
        result = results[0]

        if result.masks is not None:
            # Each entry in result.masks.data corresponds to a detected object's mask
            for box, mask_tensor in zip(result.boxes, result.masks.data):
                # Convert the mask tensor to a NumPy array
                mask = mask_tensor.cpu().numpy()
                #print(mask.shape)

                if mask.shape != annotated_image.shape[:2]:
                    mask = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                # Convert mask values from [0, 1] to [0, 255] for visualization
                mask_uint8 = (mask * 255).astype("uint8")

                # Create a blue overlay (BGR color order)
                color_mask = np.zeros_like(annotated_image)

                # ensure maks scale has correct size
                if mask_uint8.shape != color_mask.shape[:2]:
                     print(f"Warnung: Maskenform {mask_uint8.shape} passt nicht zu Bildform {color_mask.shape[:2]}. Skalierung in 'detect_obj' wird durchgeführt.")
                     mask_uint8 = cv2.resize(mask_uint8, (color_mask.shape[1], color_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                color_mask[:, :, 0] = mask_uint8  # Fill blue channel

                # Blend the mask overlay with the original image
                alpha = 0.5  # Transparency factor (0 = transparent, 1 = opaque)
                annotated_image = cv2.addWeighted(annotated_image, 1, color_mask, alpha, 0)

                # Store the mask and its corresponding class name
                obj_masks.append({
                    "class": self.model.names[int(box.cls[0])],
                    "mask": mask
                })

        if annotated_image is None:
            annotated_image=color_image.copy()
        return obj_masks, annotated_image

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------
    def fuse(self, color_image, depth_image, obj_masks):
        """
        Generates point clouds from colour and depth images for each detected object.
        Returns a dictionary:
            { label_index: {
                "label": str,
                "points": Nx3,
                "colors": Nx3,
                "median": array([X_m, Y_m, Z_m])
            } }
        """

        # Camera parameters of the depth camera (not colour camera!)
        intrinsics_depth = self.pipeline.get_active_profile().get_stream(rs.stream.depth)\
                       .as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.ppx, intrinsics_depth.ppy 

        self.cx=cx
        self.cy=cy
        self.fx=fx
        self.fy=fy

        point_cloud_results = {}

        # Initialise mask for depth image
        depth_masked = np.zeros_like(depth_image, dtype=depth_image.dtype)

        instance_counter = 0  # eindeutige ID für jede Instanz

        for obj in obj_masks:
            label = obj["class"]

            # Binary mask
            mask = (obj["mask"] > 0.5).astype(np.uint8) 

            # Mask filtering: Morphology, close/open
            if self.use_mask_filter:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            # Mask filtering: Morphology, erode, We essentially cut away the "unsafe" edge of the ball.
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.erode(mask, kernel_erode, iterations=2)

            # Adjust mask to depth resolution
            mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
            
            # Adjust colour image to depth resolution
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

            # Pixel coordinates of the mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            # Fill masked depth image
            depth_masked[ys, xs] = depth_image[ys, xs]

            # Depth values in metres
            z = depth_image[ys, xs].astype(float) * self.depth_scale
            
            # Depth filter: valid values
            valid = (z > 0.1) & (z < 10.0) & (~np.isnan(z))
            xs, ys, z = xs[valid], ys[valid], z[valid]
            if len(xs) == 0:
                continue

            # 3D coordinates (camera coordinate system)
            X = (xs - cx) * z / fx
            Y = (ys - cy) * z / fy
            #Z= np.sqrt(X**2+Y**2)
            Z = z

            # inversion for Open 3D
            Y=-Y

            points = np.stack((X, Y, Z), axis=-1)

            # Get colour values at the same pixels (BGR → RGB)
            colors = color_image[ys, xs][:, ::-1] / 255.0

             # Calculate median
            median_xyz = np.median(points, axis=0)

            # Z-Filtering
            z_median = median_xyz[2]
            z_values = points[:, 2]
            mask_z = np.abs(z_values - z_median) < 0.05 # allow only points within tolerance
            points = points[mask_z]
            colors = colors[mask_z]      
            if len(points) == 0:
                continue

            center, radius = self.fit_sphere(points)
            rmse, _ = self.sphere_fit_error(points, center, radius)
            sphere_quality = max(0, 100 * (1 - rmse / radius))

            median_xyz = np.median(points, axis=0) # recalculate median   

            point_cloud_results[instance_counter] = {
                "label": label,
                "points": points,
                "colors": colors,
                "median": median_xyz,
                "balliness": sphere_quality
            }
            instance_counter += 1


        return point_cloud_results, depth_masked
    
    def fit_sphere(points):
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
    
    def sphere_fit_error(points, center, radius):
        d = np.linalg.norm(points - center, axis=1)
        residuals = d - radius
        rmse = np.sqrt(np.mean(residuals**2))
        return rmse, residuals

    # ---------------------------------------------------------
    # Stop Camera
    # ---------------------------------------------------------
    def stop_camera(self):
        """
        Stops the RealSense pipeline.
        """
        if self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")

    def run(self):
        """
        Main loop: Read, merge and display frames.
        Exit with 'q'.
        """
        if not self.is_initialized:
            self.initialize_realsense()

        print("Starting camera stream... Press 'q' to quit.")

        # Start Open3D Visualizer once
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live 3D Point Cloud", width=900, height=700)
        pcd = o3d.geometry.PointCloud()
   
        vis.add_geometry(pcd)   # point cloud
        vis.add_geometry(axis)  # Add axes

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

                # Fuse RGB + Depth to point clouds
                pc_dict, depth_image_masked = self.fuse(color_image, depth_image, obj_masks)

                # Marked depth image
                depth_masked_normalized = cv2.normalize(depth_image_masked, None, 0,255,cv2.NORM_MINMAX)
                depth_masked_normalized= depth_masked_normalized.astype(np.uint8)
                depth_image_masked_color =cv2.applyColorMap(depth_masked_normalized, cv2.COLORMAP_JET)

                # Write median coordinates on the image
                for instance in pc_dict.values():
                    median = instance["median"]
                    balliness = instance["balliness"]
                    # Approximate pixel coordinates of the median
                    x_pixel = int((median[0] * self.fx) / median[2] + self.cx)
                    y_pixel = int((-median[1] * self.fy) / median[2] + self.cy)  # invert Y wieder für Bildkoordinaten
                    text = f"X:{median[0]:.2f} Y:{median[1]:.2f} Z:{median[2]:.2f} B:{balliness:.0f}%"
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
    test_object=ObjDetection(["Tennisball"], conf=0.5)
    test_object.run()
 