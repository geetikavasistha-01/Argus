import uuid
from datetime import datetime, timezone
from typing import List, Tuple
from pydantic import BaseModel, Field, field_validator
from shared.constants import DEFECT_CLASSES

class DetectionItem(BaseModel):
    detection_item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    class_id: int = Field(..., description="0-indexed class ID (0 to 3)")
    class_name: str = Field(..., description="Must be one of defect_class_1, defect_class_2, defect_class_3, defect_class_4")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: Tuple[float, float, float, float] = Field(..., description="[xmin, ymin, xmax, ymax]")
    mask_polygon: List[Tuple[float, float]] = Field(default_factory=list, description="List of (x, y) coordinates forming the contour boundary")
    area_px: float = Field(..., description="Estimated area of defect in square pixels")

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        if v not in DEFECT_CLASSES:
            raise ValueError(f"class_name must be one of {DEFECT_CLASSES}")
        return v

    @field_validator("class_id")
    @classmethod
    def validate_class_id(cls, v: int) -> int:
        if not (0 <= v <= 3):
            raise ValueError("class_id must be between 0 and 3 inclusive")
        return v

class InferenceResult(BaseModel):
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    image_path: str
    image_size: Tuple[int, int] = Field(..., description="[width, height]")
    detections: List[DetectionItem] = Field(default_factory=list)
    model_version: str = "yolov8m-seg-severstal-v1"
    inference_latency_ms: float
