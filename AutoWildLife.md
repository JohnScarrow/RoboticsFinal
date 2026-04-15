# AutoWildLife.py — Feature Overview

Dashcam-style real-time wildlife detection for forest roads using YOLOv8s and OpenCV.
Designed to detect animals that may cross in front of a moving vehicle.

---

## How It Differs from WildLife.py

| | WildLife.py | AutoWildLife.py |
|---|---|---|
| Use case | Stationary wildlife camera | Moving vehicle / dashcam |
| Model | MegaDetector + SpeciesNet | YOLOv8s |
| Detection speed | ~5-10 FPS | ~60-80 FPS |
| Species ID | Yes (SpeciesNet) | No (Phase 2 with custom model) |
| Close-animal warning | No | Yes |
| Custom training ready | No | Yes — one line swap |

---

## Features

### YOLOv8s Inference
Uses the YOLOv8s (small) model from Ultralytics. Significantly faster than MegaDetector
and designed for real-time video rather than static camera trap images. The pretrained
COCO model downloads automatically (~22MB) on first run. When a custom model is trained,
swapping it in requires changing one line (`MODEL_PATH`).

### Two Model Modes

**COCO Mode** (`CUSTOM_MODEL_MODE = False`) — default. Runs the pretrained YOLOv8s model
but filters detections to relevant forest road animals only: bird, cat, dog, horse, sheep,
cow, elephant, bear, zebra, giraffe. Prevents false positives from cars, furniture, people,
and the other 70+ COCO classes that aren't relevant to driving.

**Custom Model Mode** (`CUSTOM_MODEL_MODE = True`) — used once a model has been trained on
the Roboflow + LILA.science wildlife road dataset. Disables the COCO class filter and trusts
the custom model's class outputs directly (deer, elk, moose, etc.).

### Close-Animal Warning (Red Box)
When a detected animal's bounding box height is greater than 35% of the total frame height,
the box turns **red** instead of green. This indicates the animal is large in frame and
likely close to the vehicle — a higher collision risk. The threshold is tunable via
`WARNING_SIZE_THRESHOLD`.

### Confidence Threshold Filtering
Only detections above `CONFIDENCE_THRESHOLD` (default: 0.4) are shown. Filters out weak
detections that are likely noise or partial views of objects at the edge of frame.

### Webcam or Video File Input
The capture source is a single line change. Set to `0` for live webcam/dashcam feed, or
replace with a file path to run on recorded dashcam footage for testing and dataset collection.

```python
capture = cv2.VideoCapture(0)                        # live webcam
capture = cv2.VideoCapture('/path/to/dashcam.mp4')   # recorded footage
```

### Smoothed FPS Counter
Display FPS is averaged over a rolling window of 20 frames (`DISPLAY_FPS_WINDOW`). This
shows natural variation in frame rate without the number jumping on every frame. Displayed
in the top-right corner.

### Active Model Name Overlay
The name of the currently loaded model file is shown in the top-left corner of the window.
Makes it easy to confirm at a glance whether you are running the pretrained model or a
custom trained one.

---

## Configuration

All tunable values are constants at the top of [AutoWildLife.py](AutoWildLife.py):

| Constant | Default | Description |
|---|---|---|
| `MODEL_PATH` | `'yolov8s.pt'` | Model to load — swap to custom `.pt` file after training |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum confidence to show a detection |
| `CUSTOM_MODEL_MODE` | `False` | Set `True` when using a custom trained model |
| `WARNING_SIZE_THRESHOLD` | `0.35` | Fraction of frame height that triggers red warning box |
| `DISPLAY_FPS_WINDOW` | `20` | Number of frames to average FPS over |

---

## Roadmap

- **Phase 1** *(current)* — pretrained YOLOv8s on COCO, animals filtered to road-relevant classes
- **Phase 2** — custom model trained on Roboflow road/animal dataset + LILA.science wildlife images
- **Phase 3** — deploy on laptop as a live dashcam system for forest road driving

Training pipeline for Phase 2 is in `train_wildlife.py`.

---

## Running

```bash
python3 AutoWildLife.py
```

Press `q` to quit.

## Dependencies
- `ultralytics` — YOLOv8 model loading and inference
- `opencv-python` — webcam capture, drawing, display window
