"""Input-source helpers for public Google Drive files and folders."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

DRIVE_URL_PATTERN = re.compile(r"https?://(?:drive|docs)\.google\.com/[^\s<>]+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    """Return HTTP(S) URLs from a Discord command message."""
    return [url.rstrip(".,)>]}") for url in DRIVE_URL_PATTERN.findall(text)]


def is_google_drive_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"drive.google.com", "docs.google.com"}


def _image_suffix(path: Path) -> str | None:
    """Return a reliable image suffix from bytes when a download has none."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0 or cv2.imdecode(data, cv2.IMREAD_UNCHANGED) is None:
            return None
    except (OSError, ValueError):
        return None
    header = path.read_bytes()[:16]
    if header.startswith(b"\x89PNG"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return ".webp"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    return ".jpg"


def download_drive_url(url: str, destination_dir: str) -> list[str]:
    """Download a public Drive file or folder and return local source paths.

    File downloads use gdown's confirmation handling. Folder downloads use
    gdown's public-folder listing support and retain nested page directories.
    Private or login-gated links fail with a clear message rather than silently
    producing an HTML login page as if it were a chapter file.
    """
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive support requires the `gdown` package") from exc

    output_dir = Path(destination_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    path = parsed.path.lower()
    is_folder = "/folders/" in path

    if is_folder:
        folder_dir = output_dir / "drive_folder"
        folder_dir.mkdir(parents=True, exist_ok=True)
        downloaded = gdown.download_folder(
            url,
            output=str(folder_dir),
            quiet=True,
            use_cookies=False,
        )
        if not downloaded:
            raise RuntimeError(
                "The Google Drive folder could not be downloaded. Make sure it is public."
            )
        return [str(path) for path in folder_dir.rglob("*") if path.is_file()]

    filename = "drive_chapter_download"
    destination = output_dir / filename
    downloaded = gdown.download(
        url,
        output=str(destination),
        quiet=True,
        fuzzy=True,
        use_cookies=False,
    )
    if not downloaded or not Path(downloaded).is_file():
        raise RuntimeError(
            "The Google Drive file could not be downloaded. Make sure the sharing setting is public."
        )

    downloaded_path = Path(downloaded)
    if zipfile.is_zipfile(downloaded_path) and downloaded_path.suffix.lower() not in {".zip", ".cbz"}:
        normalized = downloaded_path.with_suffix(".zip")
        downloaded_path.rename(normalized)
        downloaded_path = normalized
    elif not downloaded_path.suffix:
        image_suffix = _image_suffix(downloaded_path)
        if image_suffix:
            normalized = downloaded_path.with_suffix(image_suffix)
            downloaded_path.rename(normalized)
            downloaded_path = normalized
    return [str(downloaded_path)]
