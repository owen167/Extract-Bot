"""User-facing text markers for manga/manhwa extraction output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    english: str
    arabic: str
    marker: str


LABELS: dict[str, LabelSpec] = {
    "SPEECH": LabelSpec("SPEECH", "كلام عادي", '\"\":'),
    "SYSTEM": LabelSpec("SYSTEM", "نظام", "%:"),
    "THOUGHT": LabelSpec("THOUGHT", "تفكير", "():"),
    "SHOUT": LabelSpec("SHOUT", "صراخ", "::"),
    "WHISPER": LabelSpec("WHISPER", "همس", '\"\":'),
    "SQUARE": LabelSpec("SQUARE", "فقاعة مربعة", "[]:"),
    "CAPTION": LabelSpec("CAPTION", "مربع تعليق", "[]:"),
    "NARRATION": LabelSpec("NARRATION", "سرد", "OT:"),
    "SFX": LabelSpec("SFX", "مؤثر صوتي", "SFX:"),
    "SIDE_TEXT": LabelSpec("SIDE_TEXT", "نص جانبي", "ST:"),
    "VERTICAL_TEXT": LabelSpec("VERTICAL_TEXT", "نص رأسي", "OT:"),
    "OFFSCREEN": LabelSpec("OFFSCREEN", "كلام من خارج المشهد", "OT:"),
    "RADIO": LabelSpec("RADIO", "هاتف / راديو / تلفزيون", "[]:"),
    "ELECTRIC": LabelSpec("ELECTRIC", "صوت إلكتروني / آلي", '\"\":'),
    "WAVY": LabelSpec("WAVY", "كلام متذبذب / مرتجف", '\"\":'),
    "FUZZY_THOUGHT": LabelSpec("FUZZY_THOUGHT", "تفكير ضبابي", "():"),
}

# The public Manga-Segment model currently exposes five classes. The mapping is
# deliberately configurable because some requested semantic types need a model
# trained specifically for them (for example SQUARE vs CAPTION or SFX).
DEFAULT_MODEL_LABEL_MAP: dict[str, str] = {
    "speech-balloon": "SPEECH",
    "thought-balloon": "THOUGHT",
    "caption-box": "CAPTION",
    "text": "SIDE_TEXT",
    "comic": "SYSTEM",
}


DOUBLE_MARKER_KINDS = {"SPEECH", "THOUGHT", "SQUARE", "CAPTION"}


def format_line(kind: str, text: str, sequence: int = 1) -> str:
    """Prefix OCR text with the requested marker, including paired bubbles."""
    spec = LABELS.get(kind, LABELS["SIDE_TEXT"])
    marker = "//:" if kind in DOUBLE_MARKER_KINDS and sequence >= 2 else spec.marker
    return f"{marker} {text.strip()}"


def mapping_document() -> str:
    """Return a readable English/Arabic label table for the bot README."""
    rows = ["| English | Arabic | Marker |", "|---|---|---|"]
    rows.extend(f"| {spec.english} | {spec.arabic} | `{spec.marker}` |" for spec in LABELS.values())
    return "\n".join(rows)
