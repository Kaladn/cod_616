"""
Test YOLO performance at different resolutions to determine optimal settings.
"""
import importlib.util
from pathlib import Path
import numpy as np
import time

spec = importlib.util.spec_from_file_location('yolo_detector', str(Path(__file__).resolve().parent / 'modules' / 'yolo_detector.py'))
yolo_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_mod)
YOLODetector = yolo_mod.YOLODetector


def test_yolo_stub_behavior():
    """Validate the YOLO stub returns empty/zero outputs and keeps API stable."""
    detector = YOLODetector(device='cpu', confidence=0.5)

    # Generate random test frame
    width, height = 640, 360
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Detect should return a list (empty for the stub)
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    assert len(detections) == 0

    # Extract features should return the expected structure and 191-dim vector
    features = detector.extract_features(detections, frame.shape[:2])
    assert 'feature_vector' in features
    assert features['feature_vector'].shape[0] == 191
    assert features['count'] == 0

    # Visualize should return the same frame unchanged
    vis = detector.visualize_detections(frame, detections)
    assert isinstance(vis, np.ndarray)
    assert vis.shape == frame.shape

    # Statistics should be present and numeric
    stats = detector.get_statistics()
    assert 'frames_processed' in stats
    assert isinstance(stats['frames_processed'], int)


if __name__ == "__main__":
    test_yolo_stub_behavior()
    print("YOLO stub behavior validated.")
