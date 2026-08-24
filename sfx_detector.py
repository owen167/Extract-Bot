"""Ultralytics adapter for the Hugging Face manga-sfx-detector checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedBox:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float


class SfxDetector:
    """Run object detection and expose simple pixel-space boxes."""

    def __init__(self, weights: str, *, image_size: int = 1280, confidence: float = 0.25) -> None:
        from ultralytics import YOLO

        self.image_size = image_size
        self.confidence = confidence
        self.device = "cuda:0" if self._cuda_available() else "cpu"
        self.model = YOLO(weights, task="detect")
        print(f"Loaded SFX detector: {weights} (device={self.device})")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def predict(self, image_bgr: np.ndarray) -> list[DetectedBox]:
        results = self.model.predict(
            source=image_bgr,
            stream=False,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names
        height, width = image_bgr.shape[:2]
        detected: list[DetectedBox] = []
        for index, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            detected.append(
                DetectedBox(
                    label=str(names[int(class_ids[index])]),
                    bbox=(x0, y0, x1, y1),
                    confidence=float(confidences[index]),
                )
            )
        return detected
