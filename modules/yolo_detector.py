"""
YOLO Detector (STUB)
CompuCog Multimodal Game Intelligence Engine

YOLOv8 was removed from the project due to mismatch with task goals.
This module provides a safe stub that preserves the public API and
returns empty/zero outputs where necessary to keep downstream code
working while avoiding heavy dependencies or runtime costs.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class YOLODetector:
    """Stub replacement for the original YOLODetector.

    Public API mirrors the original class but performs no inference.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        skip_frames: int = 1,
        input_resolution: Optional[Tuple[int, int]] = None
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.classes = classes
        self.skip_frames = skip_frames
        self.input_resolution = input_resolution

        # Minimal state to satisfy API
        self.frame_counter = 0
        self.last_detections: List[Dict] = []
        self.prev_detections: List[Dict] = []
        self.frames_processed = 0
        self.total_inference_time = 0.0

        print("[YOLODetector STUB] YOLO removed — using safe stub (no inference).")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Return an empty detection list quickly and deterministically."""
        self.frame_counter += 1
        self.frames_processed += 1
        self.last_detections = []
        self.prev_detections = []
        return []

    def extract_features(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Return zeroed feature arrays matching the original feature layout.

        The original implementation produced a 191-dim feature vector:
          [count(1), class_histogram(80), spatial_hist(100), area(4), conf(4), center(2)]
        """
        height, width = frame_shape
        count = len(detections)
        class_histogram = np.zeros(80, dtype=np.float32)
        spatial_histogram = np.zeros((10, 10), dtype=np.float32)

        feature_vector = np.concatenate([
            [0.0],
            class_histogram,
            spatial_histogram.flatten(),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.array([width / 2.0, height / 2.0], dtype=np.float32)
        ]).astype(np.float32)

        return {
            'count': 0,
            'class_histogram': class_histogram,
            'spatial_histogram': spatial_histogram,
            'area_stats': np.zeros(4, dtype=np.float32),
            'confidence_stats': np.zeros(4, dtype=np.float32),
            'center_mass': np.array([width / 2.0, height / 2.0], dtype=np.float32),
            'feature_vector': feature_vector
        }

    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Return the input frame unchanged (no drawing)."""
        return frame

    def get_statistics(self) -> Dict[str, float]:
        return {
            'frames_processed': self.frames_processed,
            'avg_fps': 0.0,
            'avg_inference_time_ms': 0.0
        }


if __name__ == "__main__":
    print("YOLO Detector stub — no runtime test available.")
