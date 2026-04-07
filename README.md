CS 280 Introduction to Robotics Lab
Lacey Bowden
North Idaho College
OpenCV

Today, we will be working with OpenCV (Open Source Computer Vision): an open-source
software library for computer vision, machine learning, and image processing. We will learn how
to use a webcam to take a photo, show a live photo stream, and preprocess images.
First, ensure you install the library. From your terminal, run:
pip install opencv-python
Ensure it is installed by running a python code file with OpenCV imported:
import cv2
If that runs with no errors, install was successful.

---

## Lab 6 Write-Up
**John | CS 280 | North Idaho College**

### What I Did

For this lab I used the instructor's examples (live_feed.py and motion_detector.py) to understand
how OpenCV works and then wrote my own version in main.py.

The main things I learned:

- **VideoCapture** lets you grab frames from the webcam in a loop. Each call to `capture.read()`
  gives you the next frame as a NumPy array.
- **Color space conversions** change how the image data is represented. BGR is the default in
  OpenCV (not RGB like I expected). Converting to grayscale drops it down to one channel, which
  makes it way faster to process.
- **GaussianBlur** smooths the image so small pixel-level changes (like tiny lighting shifts)
  don't trigger false motion detections.
- **Background subtraction** works by saving the very first frame and then using `absdiff()` to
  compare every new frame against it. Wherever pixels changed a lot, that's where motion happened.
- **Thresholding + dilation** cleans up the difference image into clear blobs, and then
  `connectedComponents` lets you find and measure each blob individually.
- Filtering by area (> 1000 px) ignores tiny noise blobs and only draws boxes around things
  that are actually moving in a meaningful way.

### How to Run

```bash
python main.py
```

Make sure a webcam is connected. Press `q` or close the window to quit.
When it starts, it automatically saves one preprocessed frame as `preprocessed_frame.png`.

### Deliverables

- `main.py` — Python code file
- `preprocessed_frame.png` — one preprocessed image (grayscale + Gaussian blur, saved on first frame)
