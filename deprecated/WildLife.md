# WildLife.py — Feature Overview

Live webcam wildlife detection using MegaDetector (MDV5A) and OpenCV.

## Features

### MegaDetector AI Detection
Uses the MegaDetector V5A model to detect objects in each frame. The model was trained on
camera trap images and recognizes three classes: animals, people, and vehicles. On first run
the model is downloaded automatically (~700MB) and cached locally.

### Three-Class Detection with Color-Coded Boxes
Each detected object is outlined with a bounding box colored by class:
- **Green** — animal
- **Blue** — person
- **Orange** — vehicle

### Confidence Score Labels
Each bounding box shows the class name and confidence score (e.g. `animal: 0.87`).
Only detections above the `CONFIDENCE_THRESHOLD` (default: 0.5) are displayed.

### 720p Webcam Resolution
The webcam is set to 1280x720 at startup. If the camera does not support 720p it falls
back to whatever the camera supports.

### Frame Skipping
The detector runs every `FRAME_SKIP` frames (default: 3) instead of every frame. The
display still shows every frame at full speed using the most recent detection results.
This keeps the window smooth while reducing the GPU load by ~3x.

## Configuration

These constants at the top of [WildLife.py](WildLife.py) can be changed to tune behavior:

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to show a detection |
| `FRAME_SKIP` | `3` | Run detector every N frames |

## Running

```bash
python3 WildLife.py
```

Press `q` to quit.

## Dependencies
- `opencv-python` — webcam capture, drawing, display window
- `megadetector` — AI detection model (installed from the local MegaDetector folder)
- `Pillow` — converts cv2 frames to PIL format expected by MegaDetector
