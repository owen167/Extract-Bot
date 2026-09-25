from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from sources import download_drive_url


class DriveSourceTests(unittest.TestCase):
    @patch("gdown.download")
    def test_direct_image_download_without_extension_is_normalized(self, download) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "drive_chapter_download"
            ok, encoded = cv2.imencode(".png", np.full((12, 16, 3), 255, dtype=np.uint8))
            self.assertTrue(ok)
            raw.write_bytes(encoded.tobytes())
            download.return_value = str(raw)
            result = download_drive_url("https://drive.google.com/file/d/example/view", temp)
            self.assertEqual(Path(result[0]).suffix, ".png")
            self.assertTrue(Path(result[0]).is_file())

    @patch("gdown.download_folder")
    def test_folder_download_uses_supported_arguments(self, download_folder) -> None:
        def fake_download(*args, output, **kwargs):
            page = Path(output) / "nested" / "001.png"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_bytes(b"image")
            return [str(page)]

        download_folder.side_effect = fake_download
        with tempfile.TemporaryDirectory() as temp:
            result = download_drive_url(
                "https://drive.google.com/drive/folders/example",
                temp,
            )

        self.assertEqual([Path(result[0]).name], ["001.png"])
        self.assertNotIn("remaining_ok", download_folder.call_args.kwargs)
        self.assertFalse(download_folder.call_args.kwargs["use_cookies"])


if __name__ == "__main__":
    unittest.main()
