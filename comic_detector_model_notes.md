# Comic text and bubble detector inspection

Source: https://huggingface.co/ogkalu/comic-text-and-bubble-detector

The model card identifies this as an RT-DETR-v2 r50vd detector fine-tuned on about 11k manga, webtoon, manhua, and western comic images. Training image size is 640. Classes are:

- `0: bubble`
- `1: text_bubble` (text inside bubbles)
- `2: text_free` (text outside bubbles)

The Hugging Face repository provides ONNX checkpoints, including `detector-v4-s_int8.onnx` (~11.1 MB), `detector_int8.onnx` (~43.8 MB), `detector.onnx` (~168 MB), and a Transformers `model.safetensors` (~172 MB), plus `config.json` and `preprocessor_config.json`. The repository is tagged ONNX, Safetensors, and rt_detr_v2, with Apache-2.0 license.

Integration implication: this is not a Ultralytics YOLO checkpoint. Use an ONNX Runtime adapter with the model's processor/config, or use the Transformers RT-DETR implementation. The small INT8 ONNX checkpoint is the practical CPU default. `bubble` is a container region, `text_bubble` is text inside a bubble, and `text_free` is text outside. The OCR pipeline should prefer text boxes for OCR, use bubble boxes as context, and map `text_bubble` to SPEECH by default. Unknown/ambiguous regions use the requested SPEECH marker `\"\":"`; `text_free` maps to SIDE_TEXT unless a custom mapping overrides it.
