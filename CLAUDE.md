# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install --user -r requirements.txt
```

On Raspberry Pi, swap `opencv-python` for `opencv-python-headless` and add `onnxruntime picamera2`.

Python interpreter is `/usr/bin/python3` — do not use or create a venv.

## Running Scripts

**Laptop/desktop (live webcam detection):**
```bash
python AutoWildLife.py                  # OpenCV window with live feed
python AutoWildLife.py --headless       # terminal output + MJPEG stream at http://10.42.0.1:8080
```

**Raspberry Pi (ONNX inference, no PyTorch needed):**
```bash
python detect_wildlife.py               # Pi camera live feed
python detect_wildlife.py --headless    # terminal output only
python detect_wildlife.py --source video.mp4
python detect_wildlife.py --source image.jpg
```

**Training (run on desktop with GPU):**
```bash
# Step 1 — build merged dataset + train both models
python train_wildlife.py

# Step 1 alt — RPi model only (faster, skip yolo11s)
python train_wildlife.py --rpi-only

# Step 2 — train RPi-optimised INT8 ONNX (run after train_wildlife.py)
python train_rpi.py
```

Press `q` to quit any OpenCV window. Press `Ctrl+C` to stop headless mode.

## Architecture

Two separate inference paths share the same training pipeline:

**Laptop path** (`AutoWildLife.py`): loads a `.pt` model via `ultralytics.YOLO`, runs on CUDA if available. Captures at 640×480, upscales display to 1280×720, scales bounding box coordinates proportionally. Supports a `--headless` flag that skips `cv2.imshow` and serves a live MJPEG stream over HTTP on port 8080 using Python's built-in `http.server`.

**Raspberry Pi path** (`detect_wildlife.py`): loads an INT8 ONNX model via `onnxruntime` — no PyTorch or ultralytics installed on the Pi. Uses `picamera2` for the Pi camera; falls back to `cv2.VideoCapture` for USB cameras or video files. Does its own letterbox preprocessing and YOLO output postprocessing (NMS via `cv2.dnn.NMSBoxes`). Model input is 864×480 (16:9).

**Training pipeline:**
1. `train_wildlife.py` — downloads the merged Roboflow dataset (deer, elk, turkey, moose) via a direct zip URL, downloads a LILA.science subset (Caltech Camera Traps), merges everything into `datasets/wildlife/merged/`, then trains yolo11s (laptop) and/or yolo11n (RPi base).
2. `train_rpi.py` — takes the merged dataset and trains yolo11n at 864×480, then exports FP32 ONNX and quantizes to INT8. The `best_int8.onnx` file is what gets copied to the Pi.

**Model files** are in `premadePT/` (pretrained COCO weights) and `runs/wildlife/` (custom trained weights). Neither directory is tracked by git (`*.pt`, `*.onnx`, `datasets/` are all gitignored).

**Raspberry Pi network:** Pi is at `10.42.0.1` (NetworkManager auto-IP on direct ethernet). Laptop connects at `10.42.0.2/24`. SSH: `john@10.42.0.1`.

**Early lab scripts** (`Motion Detection/`) are standalone OpenCV exercises — background subtraction using HSV V-channel diff with a 2-second frame buffer delay. `deprecated/WildLife.py` used MegaDetector + SpeciesNet and is no longer active.
