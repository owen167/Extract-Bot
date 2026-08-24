"""OCR extraction pipeline built on the Manga-Segment inference classes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from labels import DEFAULT_MODEL_LABEL_MAP, format_line

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ExtractionSettings:
    model_path: str | None = None
    comic_model_path: str | None = None
    sfx_model_path: str | None = None
    ocr_languages: str = "kor+eng+jpn"
    ocr_config: str = "--oem 1 --psm 6"
    model_confidence: float = 0.35
    sfx_confidence: float = 0.25
    comic_confidence: float = 0.25
    image_size: int = 1280
    reading_order: str = "top_to_bottom"
    min_text_length: int = 1
    model_label_map: dict[str, str] | None = None


@dataclass
class ExtractedLine:
    page: int
    region_index: int
    model_label: str
    kind: str
    text: str
    bbox: tuple[int, int, int, int]


@dataclass
class ExtractionResult:
    output_text: str
    lines: list[ExtractedLine]
    total_images: int
    failed_images: int
    elapsed_seconds: float
    output_name: str


def _load_segmenter(settings: ExtractionSettings):
    """Load the vendored Manga-Segment YOLO adapter lazily."""
    import sys

    package_root = Path(__file__).resolve().parent / "manga_segment"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    import algorithms
    from algorithms.yolo.segmenter import YoloSegmenter
    from algorithms.yolo.weights import weights_from_dir

    algorithms.load_all()
    if not settings.model_path:
        return None
    model_path = Path(settings.model_path).expanduser()
    if model_path.is_dir():
        model_path = Path(weights_from_dir(str(model_path)))
    if not model_path.is_file():
        # The RT-DETR comic detector can run as the primary detector without a
        # separate Manga-Segment checkpoint. Keep the optional legacy model silent.
        return None
    return YoloSegmenter(
        str(model_path),
        imgsz=settings.image_size,
        conf=settings.model_confidence,
        retina_masks=True,
        verbose=False,
    )


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:80] or "chapter_extracted"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def expand_inputs(input_paths: Iterable[str], work_dir: str) -> list[Path]:
    """Copy images and safely expand ZIP/CBZ attachments into a work directory."""
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []

    for raw in input_paths:
        source = Path(raw)
        if not source.is_file():
            continue
        if _is_image(source):
            destination = root / source.name
            shutil.copy2(source, destination)
            images.append(destination)
            continue
        if source.suffix.lower() not in {".zip", ".cbz"}:
            continue

        archive_dir = root / _safe_stem(source.stem)
        archive_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                destination = (archive_dir / member_path).resolve()
                if root not in destination.parents and destination != root:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        images.extend(sorted(path for path in archive_dir.rglob("*") if _is_image(path)))

    return sorted(images, key=lambda path: (path.name.lower(), str(path).lower()))


def _load_comic_detector(settings: ExtractionSettings):
    if not settings.comic_model_path:
        return None
    model_path = Path(settings.comic_model_path).expanduser()
    if not model_path.is_file():
        return None
    from comic_detector import ComicDetector

    return ComicDetector(
        str(model_path),
        image_size=640,
        confidence=settings.comic_confidence,
    )


def _load_sfx_detector(settings: ExtractionSettings):
    if not settings.sfx_model_path:
        return None
    model_path = Path(settings.sfx_model_path).expanduser()
    if not model_path.is_file():
        return None
    from sfx_detector import SfxDetector

    return SfxDetector(
        str(model_path),
        image_size=settings.image_size,
        confidence=settings.sfx_confidence,
    )


def _ocr_crop(crop_bgr: np.ndarray, languages: str, config: str) -> str:
    """Run Tesseract through pytesseract, keeping the dependency optional at import time."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is not installed; install requirements.txt") from exc

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # Upscaling helps small Korean/Japanese glyphs in webtoon screenshots.
    height, width = crop_rgb.shape[:2]
    scale = 2 if max(height, width) < 1200 else 1
    if scale > 1:
        crop_rgb = cv2.resize(crop_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(crop_rgb, lang=languages, config=config)
    return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _mask_bbox(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _masked_crop(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = image[y0:y1, x0:x1].copy()
    binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    if binary.shape != image.shape[:2]:
        binary = cv2.resize(binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    crop_mask = binary[y0:y1, x0:x1]
    # Keep the complete bubble crop; mask only removes unrelated outside content.
    outside = crop_mask == 0
    crop[outside] = 255
    return crop


def _kind_for_model_label(label: str, settings: ExtractionSettings) -> str:
    mapping = dict(DEFAULT_MODEL_LABEL_MAP)
    if settings.model_label_map:
        mapping.update(settings.model_label_map)
    # Unknown/ambiguous bubbles are deliberately treated as ordinary speech.
    return mapping.get(label, "SPEECH")


def _box_center_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


def _box_gap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    first_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    second_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _sort_lines(lines: list[ExtractedLine], order: str) -> list[ExtractedLine]:
    if order == "left_to_right":
        return sorted(lines, key=lambda line: (line.page, line.bbox[0], line.bbox[1]))
    # Webtoon/manga extraction defaults to the natural top-to-bottom page order.
    return sorted(lines, key=lambda line: (line.page, line.bbox[1], line.bbox[0]))


def extract_chapter(
    image_paths: list[str],
    settings: ExtractionSettings,
    output_name: str = "chapter_extracted",
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> ExtractionResult:
    started = time.perf_counter()
    comic_detector = _load_comic_detector(settings)
    segmenter = _load_segmenter(settings)
    sfx_detector = _load_sfx_detector(settings)
    if comic_detector is None and segmenter is None:
        raise FileNotFoundError(
            "No comic detector is available. Set COMIC_MODEL_PATH or MANGA_MODEL_PATH."
        )
    extracted: list[ExtractedLine] = []
    failed = 0

    for page_number, raw_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue
        try:
            height, width = image.shape[:2]
            page_lines: list[ExtractedLine] = []
            candidates: list[tuple[int, str, str, tuple[int, int, int, int], np.ndarray | None]] = []

            if comic_detector is not None:
                comic_detections = comic_detector.predict(image)
                text_boxes: list[tuple[int, int, int, int]] = []
                bubble_detections = []
                for index, detection in enumerate(comic_detections):
                    if detection.label == "bubble":
                        bubble_detections.append(detection)
                        continue
                    kind = {"text_bubble": "SPEECH", "text_free": "SIDE_TEXT"}.get(
                        detection.label, "SPEECH"
                    )
                    # text_free is often emitted on top of a text_bubble box by
                    # this checkpoint. Prefer the in-bubble text classification.
                    if detection.label == "text_free" and any(
                        item[1] == "text_bubble"
                        and (_box_iou(item[3], detection.bbox) >= 0.05 or _box_center_inside(detection.bbox, item[3]))
                        for item in candidates
                    ):
                        continue
                    text_boxes.append(detection.bbox)
                    candidates.append((index, detection.label, kind, detection.bbox, None))

                # If a bubble was detected but its inner text box was missed, keep
                # the bubble as ordinary dialogue instead of dropping the text.
                for index, detection in enumerate(bubble_detections, start=len(candidates)):
                    if not any(_box_iou(detection.bbox, text_box) >= 0.10 for text_box in text_boxes):
                        candidates.append((index, detection.label, "SPEECH", detection.bbox, None))
            else:
                result = segmenter.predict(image)
                for index, instance in enumerate(result.instances):
                    # `comic` describes a panel, not a text-bearing region. OCRing it
                    # would duplicate all dialogue inside the panel.
                    if instance.label == "comic":
                        continue
                    bbox = _mask_bbox(instance.mask, width, height)
                    if bbox is not None:
                        candidates.append((index, instance.label, _kind_for_model_label(instance.label, settings), bbox, instance.mask))

            # The SFX model detects sound-effect boxes. They override overlapping
            # generic text regions so sound effects are not emitted twice.
            if sfx_detector is not None:
                sfx_boxes = sfx_detector.predict(image)
                for candidate in sfx_boxes:
                    # The SFX checkpoint is intentionally conservative. Its old
                    # low threshold caused ordinary dialogue to be relabeled SFX.
                    # Never replace a comic detector text region with SFX.
                    overlaps_text = any(
                        _box_iou(item[3], candidate.bbox) >= 0.10
                        or _box_center_inside(candidate.bbox, item[3])
                        for item in candidates
                    )
                    if candidate.confidence < max(settings.sfx_confidence, 0.70) or overlaps_text:
                        continue
                    candidates.append((len(candidates), f"sfx:{candidate.label}", "SFX", candidate.bbox, None))

            for index, model_label, kind, bbox, mask in candidates:
                crop = _masked_crop(image, mask, bbox) if mask is not None else image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                text = _ocr_crop(crop, settings.ocr_languages, settings.ocr_config)
                if len(text) < settings.min_text_length:
                    continue
                page_lines.append(
                    ExtractedLine(
                        page=page_number,
                        region_index=index,
                        model_label=model_label,
                        kind=kind,
                        text=text,
                        bbox=bbox,
                    )
                )
            sorted_lines = _sort_lines(page_lines, settings.reading_order)
            extracted.extend(sorted_lines)
            if progress_callback is not None:
                progress_callback(page_number, len(image_paths), len(extracted))
        except Exception:
            failed += 1
            if progress_callback is not None:
                progress_callback(page_number, len(image_paths), len(extracted))

    pages: dict[int, list[ExtractedLine]] = {}
    for line in extracted:
        pages.setdefault(line.page, []).append(line)

    output: list[str] = []
    for page in sorted(pages):
        output.append(f"--- Page {page} ---")
        previous_line: ExtractedLine | None = None
        for line in pages[page]:
            sequence = 1
            if (
                previous_line is not None
                and previous_line.kind == line.kind
                and line.kind in {"SPEECH", "THOUGHT", "SQUARE", "CAPTION"}
                and _box_gap(previous_line.bbox, line.bbox) <= 12
            ):
                sequence = 2
            output.append(format_line(line.kind, line.text, sequence))
            previous_line = line
        output.append("")

    return ExtractionResult(
        output_text="\n".join(output).rstrip() + "\n",
        lines=extracted,
        total_images=len(image_paths),
        failed_images=failed,
        elapsed_seconds=time.perf_counter() - started,
        output_name=_safe_stem(output_name),
    )


def settings_from_env() -> ExtractionSettings:
    raw_map = os.getenv("MODEL_LABEL_MAP_JSON", "").strip()
    custom_map: dict[str, str] | None = None
    if raw_map:
        parsed = json.loads(raw_map)
        if not isinstance(parsed, dict):
            raise ValueError("MODEL_LABEL_MAP_JSON must be a JSON object")
        custom_map = {str(key): str(value) for key, value in parsed.items()}

    model_path = os.getenv("MANGA_MODEL_PATH", "").strip() or None
    comic_model_path = os.getenv(
        "COMIC_MODEL_PATH", "./models/comic-text-detector/detector-v4-s_int8.onnx"
    ).strip() or None
    sfx_model_path = os.getenv("SFX_MODEL_PATH", "./models/manga-sfx-detector.pt").strip() or None
    if not model_path and not comic_model_path:
        raise ValueError("Set COMIC_MODEL_PATH or MANGA_MODEL_PATH")
    return ExtractionSettings(
        model_path=model_path,
        comic_model_path=comic_model_path,
        sfx_model_path=sfx_model_path,
        ocr_languages=os.getenv("OCR_LANGUAGES", "kor+eng+jpn"),
        ocr_config=os.getenv("OCR_CONFIG", "--oem 1 --psm 6"),
        model_confidence=float(os.getenv("MODEL_CONFIDENCE", "0.35")),
        sfx_confidence=float(os.getenv("SFX_CONFIDENCE", "0.70")),
        comic_confidence=float(os.getenv("COMIC_CONFIDENCE", "0.25")),
        image_size=int(os.getenv("MANGA_IMAGE_SIZE", "1280")),
        reading_order=os.getenv("READING_ORDER", "top_to_bottom"),
        min_text_length=int(os.getenv("MIN_TEXT_LENGTH", "1")),
        model_label_map=custom_map,
    )
