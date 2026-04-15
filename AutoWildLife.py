# AutoWildLife.py — v1.3
#
# Dashcam-style wildlife detection for forest roads using YOLOv8s.
#
# Phase 1: Uses a pretrained YOLOv8s model (COCO dataset) so you can run
#           immediately. Detects deer, bear, bird, horse, cow, dog, cat etc.
#           Not tuned for road/forest context yet but functional out of the box.
#
# Phase 2: Swap MODEL_PATH to your custom trained model once the dataset
#           pipeline (Roboflow + LILA.science) is ready. One line change.
#
# Training pipeline is in train_wildlife.py (coming next).

import cv2
import sys
import time
from collections import deque
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------
# Phase 1: pretrained YOLOv8s on COCO — downloads automatically on first run (~22MB)
# Phase 2: swap to your custom trained model, e.g. 'runs/train/weights/best.pt'
MODEL_PATH = 'yolo11s.pt'

# Only show detections above this confidence
CONFIDENCE_THRESHOLD = 0.4

# COCO animal classes relevant to forest road driving.
# Filtering to these avoids false positives from cars, people, etc.
# Full COCO class list: https://docs.ultralytics.com/datasets/detect/coco/
ANIMAL_CLASSES = {
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
}

# Once we have a custom model trained on road wildlife, we'll detect
# species-specific classes (deer, elk, moose, etc.) instead of COCO classes.
# Set to True to disable the COCO class filter and show all detections.
CUSTOM_MODEL_MODE = False

# Box color in BGR
DETECTION_COLOR = (0, 255, 0)   # green for animals
WARNING_COLOR   = (0, 0, 255)   # red for close/large detections (high risk)

# If a bounding box covers more than this fraction of frame height,
# the animal is considered close — box turns red as a warning
WARNING_SIZE_THRESHOLD = 0.35

# -----------------------------------------------------------------------
# FPS smoothing
# -----------------------------------------------------------------------
# Smoothed over last 20 frames — enough to show natural variation
# without the number jumping every frame
DISPLAY_FPS_WINDOW = 20

# -----------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading model: {MODEL_PATH} (downloads on first run if pretrained)...')
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f'Model loaded on {DEVICE.upper()} ({torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"}). Starting webcam...')

# -----------------------------------------------------------------------
# Webcam / video input
# -----------------------------------------------------------------------
# Change 0 to a video file path to run on recorded dashcam footage instead
# e.g. cv2.VideoCapture('/path/to/dashcam.mp4')
# Capture at 640x480 (22 FPS) then upscale display to 1280x720.
# YOLOv8 resizes frames to 640px internally so detection quality is identical.
# The upscaled display is slightly softer but runs at ~22 FPS vs ~8 FPS at native 720p.
CAPTURE_W, CAPTURE_H = 640, 480
DISPLAY_W, DISPLAY_H = 1280, 720

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
cv2.namedWindow('AutoWildLife - Road Detection')

# -----------------------------------------------------------------------
# FPS tracking
# -----------------------------------------------------------------------
display_times = deque(maxlen=DISPLAY_FPS_WINDOW)

while True:
    ret, frame = capture.read()
    if not ret:
        sys.exit()

    # Upscale captured frame to display resolution for the window.
    # Inference runs on the original small frame — upscale only affects display.
    display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    h, w = DISPLAY_H, DISPLAY_W

    # -------------------------------------------------------------------
    # Inference — runs on the small captured frame (faster, same quality)
    # -------------------------------------------------------------------
    # verbose=False suppresses per-frame console output.
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # -------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------
    for box in results.boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])

        # In COCO mode, skip anything that isn't a relevant animal class.
        # In custom model mode, trust the model's classes directly.
        if not CUSTOM_MODEL_MODE and class_id not in ANIMAL_CLASSES:
            continue

        # Get label — use COCO animal name or model's own class name
        if CUSTOM_MODEL_MODE:
            label_name = model.names[class_id]
        else:
            label_name = ANIMAL_CLASSES[class_id]

        # Scale bbox coords from capture resolution up to display resolution
        scale_x = DISPLAY_W / CAPTURE_W
        scale_y = DISPLAY_H / CAPTURE_H
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Warn if animal is large in frame (likely close to the vehicle)
        bbox_height_ratio = (y2 - y1) / h
        color = WARNING_COLOR if bbox_height_ratio >= WARNING_SIZE_THRESHOLD else DETECTION_COLOR

        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------------------------------------------------
    # FPS overlay — top right
    # -------------------------------------------------------------------
    now = time.time()
    display_times.append(now)

    if len(display_times) >= 2:
        fps = (len(display_times) - 1) / (display_times[-1] - display_times[0])
    else:
        fps = 0.0

    fps_text = f"FPS: {fps:.1f}"
    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(display_frame, fps_text, (w - text_size[0] - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Model name reminder so you always know which model is running
    cv2.putText(display_frame, MODEL_PATH, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow('AutoWildLife - Road Detection', display_frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
