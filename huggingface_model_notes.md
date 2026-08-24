# Hugging Face model inspection

Source: https://huggingface.co/cozy-creator/manga-sfx-detector

The model card describes `cozy-creator/manga-sfx-detector` as a YOLO-based detector trained specifically to detect manga sound effect bubbles. The repository is tagged for object detection and yolov8, and the model file is `manga-sfx-detector.pt` (about 40.5 MB). The model is not deployed by an Inference Provider, so local Ultralytics inference is the appropriate integration path.

Integration implication: the checkpoint is an object-detection model, while the existing Manga-Segment adapter assumes segmentation masks. The bot should load this checkpoint through a dedicated detector adapter that uses `result.boxes.xyxy`, `result.boxes.cls`, and `result.names`, then crop each box for OCR. SFX detections map to `SFX`; any unknown or ambiguous detection must map to `SPEECH` and use the `\"\":"` marker.
