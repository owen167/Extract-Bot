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
from typing import Iterable

import cv2
import numpy as np

from labels import DEFAULT_MODEL_LABEL_MAP, format_line

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ExtractionSettings:
    model_path: str
    ocr_languages: str = "kor+eng+jpn"
    ocr_config: str = "--oem 1 --psm 6"
    model_confidence: float = 0.35
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
    model_path = Path(settings.model_path).expanduser()
    if model_path.is_dir():
        model_path = Path(weights_from_dir(str(model_path)))
    if not model_path.is_file():
        raise FileNotFoundError(f"MANGA_MODEL_PATH does not exist: {model_path}")
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
    return mapping.get(label, "SIDE_TEXT")


def _sort_lines(lines: list[ExtractedLine], order: str) -> list[ExtractedLine]:
    if order == "left_to_right":
        return sorted(lines, key=lambda line: (line.page, line.bbox[0], line.bbox[1]))
    # Webtoon/manga extraction defaults to the natural top-to-bottom page order.
    return sorted(lines, key=lambda line: (line.page, line.bbox[1], line.bbox[0]))


def extract_chapter(image_paths: list[str], settings: ExtractionSettings, output_name: str = "chapter_extracted") -> ExtractionResult:
    started = time.perf_counter()
    segmenter = _load_segmenter(settings)
    extracted: list[ExtractedLine] = []
    failed = 0

    for page_number, raw_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue
        try:
            result = segmenter.predict(image)
            height, width = image.shape[:2]
            page_lines: list[ExtractedLine] = []
            for index, instance in enumerate(result.instances):
                # `comic` describes a panel, not a text-bearing region. OCRing it
                # would duplicate all dialogue inside the panel.
                if instance.label == "comic":
                    continue
                bbox = _mask_bbox(instance.mask, width, height)
                if bbox is None:
                    continue
                crop = _masked_crop(image, instance.mask, bbox)
                text = _ocr_crop(crop, settings.ocr_languages, settings.ocr_config)
                if len(text) < settings.min_text_length:
                    continue
                page_lines.append(
                    ExtractedLine(
                        page=page_number,
                        region_index=index,
                        model_label=instance.label,
                        kind=_kind_for_model_label(instance.label, settings),
                        text=text,
                        bbox=bbox,
                    )
                )
            extracted.extend(_sort_lines(page_lines, settings.reading_order))
        except Exception:
            failed += 1

    pages: dict[int, list[ExtractedLine]] = {}
    for line in extracted:
        pages.setdefault(line.page, []).append(line)

    output: list[str] = []
    for page in sorted(pages):
        output.append(f"--- Page {page} ---")
        output.extend(format_line(line.kind, line.text) for line in pages[page])
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

    model_path = os.getenv("MANGA_MODEL_PATH", "").strip()
    if not model_path:
        raise ValueError("MANGA_MODEL_PATH is not configured")
    return ExtractionSettings(
        model_path=model_path,
        ocr_languages=os.getenv("OCR_LANGUAGES", "kor+eng+jpn"),
        ocr_config=os.getenv("OCR_CONFIG", "--oem 1 --psm 6"),
        model_confidence=float(os.getenv("MODEL_CONFIDENCE", "0.35")),
        image_size=int(os.getenv("MANGA_IMAGE_SIZE", "1280")),
        reading_order=os.getenv("READING_ORDER", "top_to_bottom"),
        min_text_length=int(os.getenv("MIN_TEXT_LENGTH", "1")),
        model_label_map=custom_map,
    )
