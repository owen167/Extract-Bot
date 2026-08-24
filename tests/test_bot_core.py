from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manga_segment"))

from core.segmenter import Instance, SegmentationResult  # noqa: E402
from extractor import ExtractionSettings, _is_plausible_text, extract_chapter, expand_inputs  # noqa: E402
from labels import format_line
from sources import extract_urls


class BotCoreTests(unittest.TestCase):
    def test_marker_format_matches_reference(self) -> None:
        self.assertEqual(format_line("SPEECH", "반응이 바로 오네."), '\"\": 반응이 바로 오네.')
        self.assertEqual(format_line("SYSTEM", "그 정령석을 내게 맡기게!"), "%: 그 정령석을 내게 맡기게!")
        self.assertEqual(format_line("THOUGHT", "관심 있는 정도가 아니었네."), "(): 관심 있는 정도가 아니었네.")
        self.assertEqual(format_line("SPEECH", "second bubble", 2), "//: second bubble")
        self.assertEqual(format_line("SQUARE", "second square", 2), "//: second square")

    def test_drive_url_is_extracted_from_command_text(self) -> None:
        text = "Chapter 0041 https://drive.google.com/file/d/abc123/view?usp=sharing"
        self.assertEqual(
            extract_urls(text),
            ["https://drive.google.com/file/d/abc123/view?usp=sharing"],
        )

    def test_extraction_formats_pages_and_sorts_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "001.png"
            image = np.full((120, 160, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            upper = np.zeros((120, 160), dtype=np.float32)
            upper[10:35, 20:70] = 1.0
            lower = np.zeros((120, 160), dtype=np.float32)
            lower[70:100, 30:90] = 1.0
            fake_result = SegmentationResult(
                image_bgr=image,
                instances=[
                    Instance(label="speech-balloon", mask=lower),
                    Instance(label="thought-balloon", mask=upper),
                ],
            )

            class FakeSegmenter:
                def predict(self, _image):
                    return fake_result

            settings = type("Settings", (), {
                "model_path": "unused",
                "comic_model_path": None,
                "sfx_model_path": None,
                "ocr_languages": "eng",
                "ocr_config": "",
                "model_confidence": 0.35,
                "sfx_confidence": 0.25,
                "comic_confidence": 0.25,
                "image_size": 1280,
                "reading_order": "top_to_bottom",
                "min_text_length": 1,
                "model_label_map": None,
            })()
            progress: list[tuple[int, int, int, int]] = []
            with patch("extractor._load_segmenter", return_value=FakeSegmenter()), patch(
                "extractor._ocr_crop", side_effect=["lower text", "upper text"]
            ), patch("extractor._is_plausible_text", return_value=True):
                result = extract_chapter(
                    [str(image_path)],
                    settings,
                    "chapter",
                    lambda page, total, regions, bubbles: progress.append((page, total, regions, bubbles)),
                )

            self.assertEqual(result.total_images, 1)
            self.assertEqual(result.failed_images, 0)
            self.assertIn("--- Page 1 ---", result.output_text)
            self.assertIn("(): upper text", result.output_text)
            self.assertIn('"": lower text', result.output_text)
            self.assertLess(result.output_text.index("upper text"), result.output_text.index("lower text"))
            self.assertEqual(progress, [(1, 1, 2, 2)])

    def test_ocr_quality_gate_rejects_noise_and_keeps_korean(self) -> None:
        crop = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertFalse(_is_plausible_text("0 / 404 < 고 2", crop, "SPEECH"))
        self.assertFalse(_is_plausible_text("6", crop, "SFX"))
        self.assertTrue(_is_plausible_text("안녕 English 日本語!?", crop, "SPEECH"))
        self.assertEqual(ExtractionSettings().ocr_languages, "eng+kor+jpn")

    def test_text_bubble_inherits_containing_bubble_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "001.png"
            cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))

            class FakeComicDetector:
                def predict(self, _image):
                    from comic_detector import ComicDetection
                    return [
                        ComicDetection("bubble", (10, 10, 150, 100), 0.95, "THOUGHT"),
                        ComicDetection("text_bubble", (30, 25, 130, 80), 0.95, None),
                    ]

            settings = ExtractionSettings(
                comic_model_path="unused",
                sfx_model_path=None,
                ocr_languages="kor",
                min_text_length=1,
            )
            with patch("extractor._load_comic_detector", return_value=FakeComicDetector()), patch(
                "extractor._load_segmenter", return_value=None
            ), patch("extractor._ocr_crop", return_value="생각"):
                result = extract_chapter([str(image_path)], settings, "chapter")
            self.assertEqual(len(result.lines), 1)
            self.assertEqual(result.lines[0].kind, "THOUGHT")

    def test_sfx_does_not_replace_overlapping_bubble_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "001.png"
            cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))

            class FakeComicDetector:
                def predict(self, _image):
                    from comic_detector import ComicDetection
                    return [ComicDetection("text_bubble", (10, 10, 100, 80), 0.95, None)]

            class FakeSfxDetector:
                def predict(self, _image):
                    from sfx_detector import DetectedBox
                    return [DetectedBox("manga_sfx", (20, 20, 90, 70), 0.99)]

            settings = ExtractionSettings(
                comic_model_path="unused",
                sfx_model_path="unused",
                ocr_languages="kor",
                min_text_length=1,
            )
            with patch("extractor._load_comic_detector", return_value=FakeComicDetector()), patch(
                "extractor._load_segmenter", return_value=None
            ), patch("extractor._load_sfx_detector", return_value=FakeSfxDetector()), patch(
                "extractor._ocr_crop", return_value="대화"
            ):
                result = extract_chapter([str(image_path)], settings, "chapter")
            self.assertEqual([line.kind for line in result.lines], ["SPEECH"])
            self.assertNotIn("SFX:", result.output_text)

    def test_non_adjacent_regions_never_get_pair_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "001.png"
            cv2.imwrite(str(image_path), np.zeros((220, 160, 3), dtype=np.uint8))

            class FakeComicDetector:
                def predict(self, _image):
                    from comic_detector import ComicDetection
                    return [
                        ComicDetection("text_bubble", (10, 10, 100, 40), 0.95, None),
                        ComicDetection("text_bubble", (10, 150, 100, 190), 0.95, None),
                    ]

            settings = ExtractionSettings(comic_model_path="unused", ocr_languages="kor", min_text_length=1)
            with patch("extractor._load_comic_detector", return_value=FakeComicDetector()), patch(
                "extractor._load_segmenter", return_value=None
            ), patch("extractor._ocr_crop", side_effect=["첫 줄", "둘째 줄"]):
                result = extract_chapter([str(image_path)], settings, "chapter")
            self.assertNotIn("//:", result.output_text)

    def test_zip_expansion_is_safe_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "chapter.cbz"
            with zipfile.ZipFile(archive, "w") as writer:
                writer.writestr("002.jpg", b"not-a-real-image")
                writer.writestr("001.png", b"not-a-real-image")
                writer.writestr("../escape.png", b"must-not-extract")
            paths = expand_inputs([str(archive)], str(root / "expanded"))
            self.assertEqual([path.name for path in paths], ["001.png", "002.jpg"])
            self.assertFalse((root / "escape.png").exists())


if __name__ == "__main__":
    unittest.main()
