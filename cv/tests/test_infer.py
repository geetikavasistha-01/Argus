import os
import pytest
import cv2
import numpy as np
from cv.schemas import InferenceResult, DetectionItem
from cv.infer import run_inference

# Define file paths
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_images"))
BLANK_IMAGE_PATH = os.path.join(SAMPLE_DIR, "blank.jpg")
DEFECT_IMAGE_PATH = os.path.join(SAMPLE_DIR, "mock_defect.jpg")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "severstal_yolov8m_seg_best.pt"))

@pytest.fixture(scope="session", autouse=True)
def setup_sample_images():
    """
    Creates sample images for testing if they do not exist.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # 1. Create a blank image (1600x256, gray)
    if not os.path.exists(BLANK_IMAGE_PATH):
        blank_img = np.ones((256, 1600, 3), dtype=np.uint8) * 128
        cv2.imwrite(BLANK_IMAGE_PATH, blank_img)
        
    # 2. Create a mock defect image (1600x256, gray with a dark rectangle)
    if not os.path.exists(DEFECT_IMAGE_PATH):
        defect_img = np.ones((256, 1600, 3), dtype=np.uint8) * 128
        # Draw a dark block representing a surface defect
        cv2.rectangle(defect_img, (200, 50), (400, 200), (30, 30, 30), -1)
        cv2.imwrite(DEFECT_IMAGE_PATH, defect_img)

def test_schema_serialization():
    """
    Verifies that the InferenceResult schema can serialize and deserialize correctly.
    """
    detection = DetectionItem(
        class_id=1,
        class_name="defect_class_2",
        confidence=0.89,
        bbox_xyxy=(100.0, 50.0, 300.0, 200.0),
        mask_polygon=[(100.0, 50.0), (300.0, 50.0), (300.0, 200.0), (100.0, 200.0)],
        area_px=30000.0
    )
    
    result = InferenceResult(
        image_path="test.jpg",
        image_size=(1600, 256),
        detections=[detection],
        inference_latency_ms=12.5
    )
    
    # Convert to JSON
    json_data = result.model_dump_json()
    assert isinstance(json_data, str)
    
    # Load from JSON
    parsed = InferenceResult.model_validate_json(json_data)
    assert parsed.image_path == "test.jpg"
    assert len(parsed.detections) == 1
    assert parsed.detections[0].class_name == "defect_class_2"
    assert parsed.detections[0].bbox_xyxy == (100.0, 50.0, 300.0, 200.0)

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model weights not found. Skipping live inference tests.")
def test_infer_on_sample_image():
    """
    Runs actual inference on the mock defect image using the trained model weights.
    Only runs if the model weights are present.
    """
    result = run_inference(DEFECT_IMAGE_PATH, model_path=MODEL_PATH)
    
    # Verify result conforms to the Pydantic schema structure
    assert isinstance(result, InferenceResult)
    assert result.image_path == DEFECT_IMAGE_PATH
    assert result.image_size == (1600, 256)
    assert isinstance(result.detections, list)
    
    for det in result.detections:
        assert isinstance(det, DetectionItem)
        assert det.class_name in ["defect_class_1", "defect_class_2", "defect_class_3", "defect_class_4"]
        assert len(det.bbox_xyxy) == 4
        assert det.confidence >= 0.0 and det.confidence <= 1.0

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="YOLO model weights not found. Skipping live inference tests.")
def test_no_false_positives_on_blank():
    """
    Verifies that a blank image yields no or very low confidence detections.
    """
    result = run_inference(BLANK_IMAGE_PATH, model_path=MODEL_PATH, conf_threshold=0.5)
    assert isinstance(result, InferenceResult)
    # On a blank image, we expect no defects detected with high confidence
    assert len(result.detections) == 0
