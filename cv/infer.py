import os
import time
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from ultralytics import YOLO
from cv.schemas import InferenceResult, DetectionItem
from shared.constants import DEFECT_CLASSES

# Global cache to keep the model loaded in memory across calls (FR-1.4 / API design)
_MODEL_CACHE = {}

def get_model(model_path: str) -> YOLO:
    if model_path not in _MODEL_CACHE:
        print(f"Loading YOLO model weights from {model_path} into memory...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO weights not found at {model_path}")
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]

def run_inference(image_path: str, model_path: str = "models/severstal_yolov8m_seg_best.pt", conf_threshold: float = 0.3) -> InferenceResult:
    """
    Runs YOLOv8-seg inference on an image and returns the typed InferenceResult schema.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Read image to get original dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    h, w = img.shape[:2]
    
    # Load model and run inference
    model = get_model(model_path)
    
    start_time = time.perf_counter()
    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    detections = []
    
    if len(results) > 0:
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            
            # Map index to class name
            if 0 <= class_id < len(DEFECT_CLASSES):
                class_name = DEFECT_CLASSES[class_id]
            else:
                class_name = f"unknown_class_{class_id}"
                
            # Bounding box xyxy format
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            
            # Mask polygon coords (normalized to 0-1) and pixel area
            mask_poly = []
            area_px = 0.0
            
            if masks is not None and len(masks.xy) > i:
                # Contours in original pixel space
                pixel_contour = masks.xy[i]
                if len(pixel_contour) >= 3:
                    area_px = float(cv2.contourArea(pixel_contour.astype(np.float32)))
                    # Normalize coords for polygon schema
                    mask_poly = [(float(pt[0]), float(pt[1])) for pt in pixel_contour]
            
            detection_item = DetectionItem(
                detection_item_id=str(uuid.uuid4()),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                mask_polygon=mask_poly,
                area_px=area_px
            )
            detections.append(detection_item)
            
    return InferenceResult(
        detection_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        image_path=image_path,
        image_size=(w, h),
        detections=detections,
        model_version=f"yolov8-seg-{os.path.basename(model_path)}",
        inference_latency_ms=latency_ms
    )
