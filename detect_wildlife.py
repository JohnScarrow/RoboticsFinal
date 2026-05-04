# detect_wildlife.py
#
# Wildlife detection inference script for Raspberry Pi 5.
# Runs the INT8 ONNX model using onnxruntime — no PyTorch or ultralytics needed.
#
# Usage:
#   python3 detect_wildlife.py                        # USB/Pi camera live feed
#   python3 detect_wildlife.py --source video.mp4     # video file
#   python3 detect_wildlife.py --source image.jpg     # single image
#   python3 detect_wildlife.py --camera 0             # choose camera index
#
# Controls:
#   q — quit

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
MODEL_PATH  = Path(__file__).parent / 'best_int8.onnx'
CLASS_NAMES = ['0', 'Deer', 'Roe deer', 'deers', 'doe', 'elk', 'roedeer', 'waterdeer']

CONF_THRESH = 0.35   # minimum confidence to show a detection
IOU_THRESH  = 0.45   # NMS overlap threshold

# Model input size — must match what train_rpi.py used (864×480, 16:9)
INPUT_W = 864
INPUT_H = 480

# Colour per class (BGR)
COLORS = [
    (0,   255, 100),   # 0
    (0,   200, 255),   # Deer
    (0,   150, 255),   # Roe deer
    (50,  255, 50 ),   # deers
    (255, 100, 0  ),   # doe
    (255, 200, 0  ),   # elk
    (200, 0,   255),   # roedeer
    (0,   100, 255),   # waterdeer
]

# -----------------------------------------------------------------------
# Pre/post processing helpers
# -----------------------------------------------------------------------
def letterbox(img, target_w, target_h):
    """Resize with padding to preserve aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_h, target_h - new_h - pad_h,
                                          pad_w, target_w - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, scale, pad_w, pad_h


def preprocess(frame):
    """BGR frame → NCHW float32 blob."""
    img, scale, pad_w, pad_h = letterbox(frame, INPUT_W, INPUT_H)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # CHW → NCHW
    return img, scale, pad_w, pad_h


def postprocess(output, scale, pad_w, pad_h, orig_w, orig_h):
    """
    YOLOv11 ONNX output shape: [1, 4+nc, 8400]
    Rows 0-3: cx, cy, w, h (in input-image pixels)
    Rows 4+:  class scores
    """
    preds = output[0]                            # [4+nc, 8400]
    preds = np.transpose(preds, (1, 0))          # [8400, 4+nc]

    boxes_raw  = preds[:, :4]
    scores_raw = preds[:, 4:]

    class_ids  = np.argmax(scores_raw, axis=1)
    confidences = scores_raw[np.arange(len(scores_raw)), class_ids]

    mask = confidences >= CONF_THRESH
    boxes_raw   = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []

    # cx,cy,w,h → x1,y1,x2,y2 (still in letterboxed input coords)
    cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Remove letterbox padding, undo scale → original frame coords
    x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
    y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
    x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
    y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(int)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), CONF_THRESH, IOU_THRESH
    )
    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()
    return boxes_xyxy[indices], confidences[indices], class_ids[indices]


def draw(frame, boxes, confidences, class_ids):
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls_id % len(COLORS)]
        label = f'{CLASS_NAMES[cls_id]} {conf:.0%}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Wildlife detection — RPi5')
    parser.add_argument('--source',   default=None,           help='Image or video file (default: camera)')
    parser.add_argument('--camera',   type=int, default=0,    help='Camera index (default: 0)')
    parser.add_argument('--model',    default=str(MODEL_PATH),help='Path to ONNX model')
    parser.add_argument('--headless', action='store_true',    help='No window — print detections to terminal')
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f'ERROR: Model not found at {model_path}')
        print('Copy best_int8.onnx to the same folder as this script.')
        raise SystemExit(1)

    print(f'Loading model: {model_path}')
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4   # use all 4 RPi5 cores
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=['CPUExecutionProvider'],
    )
    input_name = session.get_inputs()[0].name
    print('Model loaded.')

    # Open source
    if args.source:
        src = args.source
        is_image = Path(src).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    else:
        src = args.camera
        is_image = False

    if is_image:
        cap = None
        picam2 = None
    elif args.source:
        cap = cv2.VideoCapture(src)
        picam2 = None
        if not cap.isOpened():
            print(f'ERROR: Could not open video: {src}')
            raise SystemExit(1)
    else:
        cap = None
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (INPUT_W, INPUT_H), "format": "BGR888"}
        ))
        picam2.start()

    print('Running — press Ctrl+C to stop.' if args.headless else 'Running — press q to quit.')
    fps_counter, fps_start = 0, time.time()
    fps_display = 0.0

    while True:
        if is_image:
            frame = cv2.imread(args.source)
            if frame is None:
                print(f'ERROR: Could not read image: {args.source}')
                break
        elif picam2:
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        orig_h, orig_w = frame.shape[:2]

        # Inference
        blob, scale, pad_w, pad_h = preprocess(frame)
        t0 = time.time()
        outputs = session.run(None, {input_name: blob})
        inference_ms = (time.time() - t0) * 1000

        boxes, confidences, class_ids = postprocess(
            outputs[0], scale, pad_w, pad_h, orig_w, orig_h
        )

        # FPS tracking
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter / (time.time() - fps_start)
            fps_counter, fps_start = 0, time.time()

        if args.headless:
            ts = datetime.now().strftime('%H:%M:%S')
            if boxes is not None and len(boxes):
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    print(f'[{ts}] {CLASS_NAMES[cls_id]} {conf:.0%}  box=({x1},{y1},{x2},{y2})  {inference_ms:.0f}ms')
            else:
                print(f'[{ts}] no detection  {fps_display:.1f}FPS  {inference_ms:.0f}ms')
        else:
            frame = draw(frame, boxes, confidences, class_ids)
            cv2.putText(frame, f'{fps_display:.1f} FPS  |  {inference_ms:.0f} ms',
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('AutoWildLife', frame)

            if is_image:
                cv2.waitKey(0)
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap:
        cap.release()
    if picam2:
        picam2.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
