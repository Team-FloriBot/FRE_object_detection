import sys
import os
from ultralytics import YOLO

def export_model():
    # Get model path from environment variable or use default
    model_path = os.getenv("MODEL_PATH", "/model/yolo26n_jute_stripe_yellow_paper_02-seg.pt")
    device_suffix = os.getenv("DEVICE_SUFFIX", "") # e.g. "_pc" or "_jetson"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting model to TensorRT (engine)...")
    # Export to engine format.
    exported_path = model.export(format="engine", imgsz=640, dynamic=False, simplify=True)
    
    # Rename if suffix is provided
    if device_suffix and exported_path:
        # model.export returns the path to the folder or file
        # For engine, it's usually the .engine file itself
        old_path = exported_path
        if isinstance(old_path, list): # handle list return
            old_path = old_path[0]
            
        new_path = str(old_path).replace(".engine", f"{device_suffix}.engine")
        
        print(f"Renaming {old_path} to {new_path}")
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(old_path, new_path)
    
    print("Export finished.")

if __name__ == "__main__":
    export_model()
  