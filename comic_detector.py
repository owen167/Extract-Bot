"""ONNX Runtime adapter for ogkalu/comic-text-and-bubble-detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bubble_classifier import classify_bubble_shape


@dataclass(frozen=True)
class ComicDetection:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    bubble_shape: str | None = None


class ComicDetector:
    """Run the RT-DETR-v2 ONNX checkpoint on square tiles of long comic pages."""

    LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}

    def __init__(
        self,
        weights: str,
        *,
        image_size: int = 640,
        confidence: float = 0.25,
        tile_overlap: float = 0.15,
    ) -> None:
        import onnxruntime as ort

        self.image_size = image_size
        self.confidence = confidence
        self.tile_overlap = max(0.0, min(0.4, tile_overlap))
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

    def _predict_tile(self, tile_bgr: np.ndarray, offset_y: int) -> list[ComicDetection]:
        tensor, target_sizes = self._input(tile_bgr)
        outputs = self.session.run(
            self.output_names,
            {self.input_name: tensor, self.target_sizes_name: target_sizes},
        )
        by_name = dict(zip(self.output_names, outputs))
        labels = np.asarray(by_name["labels"])[0]
        boxes = np.asarray(by_name["boxes"])[0]
        scores = np.asarray(by_name["scores"])[0]
        tile_height, tile_width = tile_bgr.shape[:2]
        detections: list[ComicDetection] = []
        for label_id, box, score in zip(labels, boxes, scores):
            score = float(score)
            if score < self.confidence:
                continue
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            x0 = max(0, min(x0, tile_width - 1))
            y0 = max(0, min(y0, tile_height - 1))
            x1 = max(x0 + 1, min(x1, tile_width))
            y1 = max(y0 + 1, min(y1, tile_height))
            detections.append(
                ComicDetection(
                    label=self.LABELS.get(int(label_id), "unknown"),
                    bbox=(x0, y0 + offset_y, x1, y1 + offset_y),
                    confidence=score,
                )
            )
        return detections

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = first
        bx0, by0, bx1, by1 = second
        intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def predict(self, image_bgr: np.ndarray) -> list[ComicDetection]:
        height, width = image_bgr.shape[:2]
        tile_height = max(width, 640)
        if height <= tile_height * 1.25:
            detections = self._predict_tile(image_bgr, 0)
        else:
            stride = max(1, int(tile_height * (1.0 - self.tile_overlap)))
            detections = []
            starts = list(range(0, max(1, height - tile_height + 1), stride))
            last_start = max(0, height - tile_height)
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                end = min(height, start + tile_height)
                detections.extend(self._predict_tile(image_bgr[start:end], start))

        # Collapse detections duplicated by overlapping tiles, keeping the stronger one.
        deduped: list[ComicDetection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                detection.label == existing.label and self._iou(detection.bbox, existing.bbox) >= 0.55
                for existing in deduped
            ):
                continue
            deduped.append(detection)
        annotated: list[ComicDetection] = []
        for detection in deduped:
            if detection.label == "bubble":
                x0, y0, x1, y1 = detection.bbox
                bubble_crop = image_bgr[y0:y1, x0:x1]
                annotated.append(
                    ComicDetection(
                        label=detection.label,
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        bubble_shape=classify_bubble_shape(bubble_crop),
                    )
                )
            else:
                annotated.append(detection)
        return sorted(annotated, key=lambda item: (item.bbox[1], item.bbox[0]))
