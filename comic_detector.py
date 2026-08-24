"""ONNX Runtime adapter for ogkalu/comic-text-and-bubble-detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ComicDetection:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float


class ComicDetector:
    """Run the RT-DETR-v2 ONNX checkpoint at its 640x640 training size."""

    LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}

    def __init__(self, weights: str, *, image_size: int = 640, confidence: float = 0.25) -> None:
        import onnxruntime as ort

        self.image_size = image_size
        self.confidence = confidence
        self.session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.target_sizes_name = self.session.get_inputs()[1].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Loaded comic detector: {weights} (provider=CPUExecutionProvider)")

    def _input(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        target_sizes = np.array([[height, width]], dtype=np.int64)
        return tensor, target_sizes

    def predict(self, image_bgr: np.ndarray) -> list[ComicDetection]:
        tensor, target_sizes = self._input(image_bgr)
        outputs = self.session.run(
            self.output_names,
            {self.input_name: tensor, self.target_sizes_name: target_sizes},
        )
        by_name = dict(zip(self.output_names, outputs))
        labels = np.asarray(by_name["labels"])[0]
        boxes = np.asarray(by_name["boxes"])[0]
        scores = np.asarray(by_name["scores"])[0]
        height, width = image_bgr.shape[:2]
        detections: list[ComicDetection] = []
        for label_id, box, score in zip(labels, boxes, scores):
            score = float(score)
            if score < self.confidence:
                continue
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            detections.append(
                ComicDetection(
                    label=self.LABELS.get(int(label_id), "unknown"),
                    bbox=(x0, y0, x1, y1),
                    confidence=score,
                )
            )
        return detections
