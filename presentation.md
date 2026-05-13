# Wildlife Detection System — CS-280 Final Project

**John Scarrow**

---

## Problem

Wildlife near roads and trails goes undetected until it's too late. Existing solutions require cloud connectivity or expensive hardware. This project builds a low-cost, fully offline wildlife detector that runs on a Raspberry Pi 5.

---

## System Overview

Two inference paths share one training pipeline:

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| **Model** | YOLOv11s (.pt) | YOLOv11n INT8 ONNX |
| **Runtime** | PyTorch / Ultralytics | ONNX Runtime (CPU only) |
| **Camera** | USB webcam | Pi Camera Module |
| **Input size** | 640×480 | 864×480 (16:9) |
| **Output** | OpenCV window or MJPEG stream | OpenCV window or terminal |

---

## Training Pipeline

1. **Dataset merging** (`train_wildlife.py`)
   - Roboflow dataset: deer, elk, turkey, moose (~4 classes, ~3,000 images)
   - LILA.science subset: Caltech Camera Traps (real-world trap footage)
   - Merged into a single `datasets/wildlife/merged/` structure

2. **Base training** (`train_wildlife.py`)
   - YOLOv11s → laptop deployment
   - YOLOv11n → Pi base weights

3. **Pi-optimised export** (`train_rpi.py`)
   - Fine-tunes YOLOv11n at 864×480
   - Exports FP32 ONNX, then quantizes to INT8
   - Result: `best_int8.onnx` — copied to the Pi, no PyTorch needed

---

## Raspberry Pi Inference

`detect_wildlife.py` does everything without PyTorch or Ultralytics:

- **Letterbox preprocessing** — pads to 864×480, preserves aspect ratio
- **ONNX Runtime inference** — 4-thread CPU, INT8 quantized model
- **Custom postprocessing** — decodes YOLO output tensor, runs NMS via `cv2.dnn.NMSBoxes`
- **Headless mode** — skips all rendering; prints timestamped detections to terminal for logging

```
[14:32:07] Deer 91%  box=(120,88,430,390)  3.2FPS  312ms
[14:32:08] Elk  78%  box=(50,200,310,480)  3.1FPS  318ms
```

---

## Network Setup

Pi is connected directly to the laptop via ethernet:

- Pi: `10.42.0.1` (NetworkManager auto-IP)
- Laptop: `10.42.0.2/24`
- SSH: `john@10.42.0.1`
- Laptop headless mode streams MJPEG at `http://10.42.0.1:8080`

---

## Results

| Metric | Value |
|---|---|
| Inference speed (Pi 5, INT8) | ~3–4 FPS |
| Inference speed (laptop, CUDA) | ~30 FPS |
| Confidence threshold | 0.35 |
| NMS IoU threshold | 0.45 |
| Classes detected | Deer, Roe deer, Doe, Elk, Moose, Turkey |

---

## Demo

- Live webcam on laptop: `python AutoWildLife.py`
- Pi camera live feed: `python detect_wildlife.py`
- Single image test: `python detect_wildlife.py --source image.jpg`

---

## Challenges

- **No PyTorch on Pi** — required implementing YOLO pre/postprocessing from scratch in NumPy/OpenCV
- **INT8 quantization** — needed careful calibration to avoid accuracy loss
- **Dataset noise** — Roboflow labels had inconsistent class names (`deers`, `roedeer`, etc.) requiring merge-time normalization

---

## Future Work

- NNAPI / CoreML delegate for hardware-accelerated inference on Pi
- GPS tagging of detections for wildlife mapping
- Solar-powered enclosure for unattended field deployment
- Alert system (SMS/email) on detection
