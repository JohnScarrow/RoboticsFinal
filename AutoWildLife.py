# AutoWildLife.py — v1.4
#
# Dashcam-style wildlife detection for forest roads using YOLOv11s.
#
# Normal mode:   python AutoWildLife.py
#                Opens an OpenCV window with live annotated feed.
#
# Headless mode: python AutoWildLife.py --headless
#                No window. Prints detections to terminal with timestamp.
#                Serves live MJPEG stream at http://<pi-ip>:8080
#                Open that URL in a browser on your laptop to see the feed.

import cv2
import sys
import time
import argparse
import threading
import datetime
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true', help='Run without display window, serve MJPEG stream on port 8080')
parser.add_argument('--port', type=int, default=8080, help='Port for MJPEG stream (headless mode only)')
args = parser.parse_args()

HEADLESS    = args.headless
STREAM_PORT = args.port

# -----------------------------------------------------------------------
# Model config
# -----------------------------------------------------------------------
MODEL_PATH = 'yolo11s.pt'

CONFIDENCE_THRESHOLD = 0.4

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

CUSTOM_MODEL_MODE = False

DETECTION_COLOR = (0, 255, 0)
WARNING_COLOR   = (0, 0, 255)
WARNING_SIZE_THRESHOLD = 0.35
DISPLAY_FPS_WINDOW     = 20

# -----------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading model: {MODEL_PATH}...')
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f'Model loaded on {DEVICE.upper()}.')

# -----------------------------------------------------------------------
# MJPEG stream server (headless mode only)
# -----------------------------------------------------------------------
# latest_jpeg holds the most recent annotated frame as a JPEG byte string.
# The HTTP handler reads it on every request chunk — no frame queue needed.
latest_jpeg = None
jpeg_lock   = threading.Lock()

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence per-request access logs

    def do_GET(self):
        if self.path != '/':
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with jpeg_lock:
                    frame = latest_jpeg
                if frame is None:
                    time.sleep(0.01)
                    continue
                self.wfile.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
        except (BrokenPipeError, ConnectionResetError):
            pass

def start_stream_server(port):
    server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
    print(f'MJPEG stream: http://192.168.2.2:{port}  (open in browser on your laptop)')
    server.serve_forever()

if HEADLESS:
    t = threading.Thread(target=start_stream_server, args=(STREAM_PORT,), daemon=True)
    t.start()

# -----------------------------------------------------------------------
# Webcam input
# -----------------------------------------------------------------------
CAPTURE_W, CAPTURE_H = 640, 480
DISPLAY_W, DISPLAY_H = 1280, 720

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

if not HEADLESS:
    cv2.namedWindow('AutoWildLife - Road Detection')

# -----------------------------------------------------------------------
# FPS tracking
# -----------------------------------------------------------------------
display_times = deque(maxlen=DISPLAY_FPS_WINDOW)

print('Running. Press Ctrl+C to stop.' if HEADLESS else 'Running. Press q to quit.')

while True:
    ret, frame = capture.read()
    if not ret:
        sys.exit()

    display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    h, w = DISPLAY_H, DISPLAY_W

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # -------------------------------------------------------------------
    # Draw boxes + collect detections for terminal output
    # -------------------------------------------------------------------
    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        conf     = float(box.conf[0])

        if not CUSTOM_MODEL_MODE and class_id not in ANIMAL_CLASSES:
            continue

        label_name = model.names[class_id] if CUSTOM_MODEL_MODE else ANIMAL_CLASSES[class_id]

        scale_x = DISPLAY_W / CAPTURE_W
        scale_y = DISPLAY_H / CAPTURE_H
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        bbox_height_ratio = (y2 - y1) / h
        close  = bbox_height_ratio >= WARNING_SIZE_THRESHOLD
        color  = WARNING_COLOR if close else DETECTION_COLOR

        label = f"{label_name} {conf:.2f}"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append((label_name, conf, close))

    # -------------------------------------------------------------------
    # Terminal output (headless mode)
    # -------------------------------------------------------------------
    if HEADLESS and detections:
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        for name, conf, close in detections:
            warning = ' — WARNING: close' if close else ''
            print(f'[{ts}] {name} {conf:.2f}{warning}')

    # -------------------------------------------------------------------
    # FPS overlay
    # -------------------------------------------------------------------
    now = time.time()
    display_times.append(now)
    fps = (len(display_times) - 1) / (display_times[-1] - display_times[0]) if len(display_times) >= 2 else 0.0

    fps_text  = f"FPS: {fps:.1f}"
    mode_text = f"{'HEADLESS' if HEADLESS else MODEL_PATH}"
    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(display_frame, fps_text,  (w - text_size[0] - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, mode_text, (10, 24),                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # -------------------------------------------------------------------
    # Output — window or MJPEG stream
    # -------------------------------------------------------------------
    if HEADLESS:
        _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with jpeg_lock:
            latest_jpeg = jpeg.tobytes()
    else:
        cv2.imshow('AutoWildLife - Road Detection', display_frame)
        if cv2.waitKey(1) == ord('q'):
            break

capture.release()
if not HEADLESS:
    cv2.destroyAllWindows()
