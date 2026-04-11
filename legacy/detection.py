"""Legacy YOLO-only entrypoint that forwards to the unified implementation."""

from ros2_detection.ros2_detection.detection_model_selection import ObjDetection as _UnifiedObjDetection


class ObjDetection(_UnifiedObjDetection):
    def __init__(
        self,
        classes,
        use_decimation=False,
        use_spatial=True,
        use_temporal=False,
        use_hole_filling=True,
        use_mask_filter=True,
        conf=0.5,
        model_path=None,
    ):
        super().__init__(
            classes=classes,
            model_type="yolo",
            model_path=model_path,
            use_decimation=use_decimation,
            use_spatial=use_spatial,
            use_temporal=use_temporal,
            use_hole_filling=use_hole_filling,
            use_mask_filter=use_mask_filter,
            conf=conf,
        )
