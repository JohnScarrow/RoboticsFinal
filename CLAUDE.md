# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install opencv-python
```

Requires a webcam for any script that uses live video capture.

## Running Scripts

```bash
python live_feed.py
python motion_detector.py
python "lab6(Class_Example).py"
```

Press `q` or close the window to exit any running script.

## Architecture

This is a CS 280 (Intro to Robotics) lab at North Idaho College focused on OpenCV image processing. There is no build system — scripts are standalone.

**lab6(Class_Example).py** — static image processing demo. Reads a `jellyfish.jpg` file and demonstrates color space conversions (BGR → Grayscale → HSV) and pixel-level access.

**live_feed.py** — webcam capture loop. Converts each frame to HSV and displays it in real time.

**motion_detector.py** — background subtraction pipeline:
1. Captures first frame as static background reference
2. Each subsequent frame is grayscaled and blurred (21×21 Gaussian)
3. Absolute difference from background → binary threshold (value: 25) → dilation (2 iterations)
4. Finds connected components; draws bounding boxes around regions with area > 1000 px

**main.py** — currently empty entry point.
