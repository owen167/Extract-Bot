# Extract Bot

A Discord bot that uses the **Manga-Segment** computer-vision project to detect manga/manhwa text regions and Tesseract OCR to extract the text into a chapter `.txt` file.

All bot-facing messages are in English and use Discord Embeds. The final result mentions the requester, sends the generated text file, and deletes the original command message after processing is complete.

## Command

Attach page images or one `.zip`/`.cbz` chapter archive and send:

```text
!extract
```

You can optionally provide an output name:

```text
!extract Chapter 0041
```

Supported page formats are PNG, JPG, JPEG, WEBP, BMP, TIF, and TIFF. The bot preserves page sections in this format:

```text
--- Page 1 ---
%: 그 정령석을 내게 맡기게!
"": 반응이 바로 오네.
[]: 비어 있는 정령석을 정령의 힘으로 채우는 건 쉬운 일이 아니다.
```

## Label markers

| English | Arabic | Marker |
|---|---|---|
| SPEECH | كلام عادي | `"":` |
| SYSTEM | نظام | `%:` |
| THOUGHT | تفكير | `():` |
| SHOUT | صراخ | `::` |
| WHISPER | همس | `"":` |
| SQUARE | فقاعة مربعة | `[]:` |
| CAPTION | مربع تعليق | `[]:` |
| NARRATION | سرد | `OT:` |
| SFX | مؤثر صوتي | `SFX:` |
| SIDE_TEXT | نص جانبي | `ST:` |
| VERTICAL_TEXT | نص رأسي | `OT:` |
| OFFSCREEN | كلام من خارج المشهد | `OT:` |
| RADIO | هاتف / راديو / تلفزيون | `[]:` |
| ELECTRIC | صوت إلكتروني / آلي | `"":` |
| WAVY | كلام متذبذب / مرتجف | `"":` |
| FUZZY_THOUGHT | تفكير ضبابي | `():` |

The current public Manga-Segment model exposes `speech-balloon`, `thought-balloon`, `caption-box`, `text`, and `comic`. These are mapped by default in `labels.py`; `MODEL_LABEL_MAP_JSON` can override the mapping when a more specialized checkpoint is supplied.

## Installation

Use Python 3.12 or newer, install Tesseract and the OCR language packs, then install Python dependencies:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor tesseract-ocr-jpn
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `DISCORD_TOKEN` and `MANGA_MODEL_PATH` in `.env`. The `.env` file is ignored by Git and must never be committed. The segmentation checkpoint is also ignored; store it locally or provide it through your deployment's secret/file storage.

Run the bot:

```bash
python bot.py
```

The Discord application must have the **Message Content Intent** enabled. The bot also needs permission to read messages, view channels, send messages, attach files, and manage messages so it can delete the original extraction command after completion.

## Architecture

The `manga_segment/` directory contains the reusable inference portion of the upstream Manga-Segment project, retained with its license in `MANGA_SEGMENT_LICENSE`. `extractor.py` loads its YOLO segmenter, computes safe bounding boxes from detected masks, crops each region, runs OCR, sorts regions in reading order, and writes the marker-prefixed chapter text.

`bot.py` handles Discord attachments and Embeds. Extraction is performed in a worker thread and guarded by a lock so a single model instance is not used concurrently by multiple requests. Temporary files are deleted after each request, including on errors.

## Security

Never commit `.env`, Discord tokens, model checkpoints, or generated chapter files. If a token is ever exposed publicly, regenerate it in the Discord Developer Portal before using the bot again.
