# WildLife.py — v1.1
import cv2
import sys
import time
from collections import deque
from PIL import Image
from speciesnet import SpeciesNetClassifier, DEFAULT_MODEL
from speciesnet.utils import BBox
from megadetector.detection.run_detector import load_detector

# -----------------------------------------------------------------------
# Thresholds
# -----------------------------------------------------------------------
# Minimum MegaDetector confidence to show any bounding box at all
CONFIDENCE_THRESHOLD = 0.5

# Minimum MegaDetector confidence required before we bother running
# SpeciesNet on an animal crop. Lower-confidence detections just show
# "ANIMAL" without species ID since they probably aren't reliable enough.
SPECIES_THRESHOLD = 0.7

# -----------------------------------------------------------------------
# Label / color config
# -----------------------------------------------------------------------
# MegaDetector outputs category IDs '1', '2', '3'
LABEL_MAP = {
    '1': 'ANIMAL',
    '2': 'PERSON',
    '3': 'VEHICLE'
}

# Box colors per class in BGR format
COLORS = {
    '1': (0, 255, 0),    # animal  - green
    '2': (255, 0, 0),    # person  - blue
    '3': (0, 165, 255)   # vehicle - orange
}

# -----------------------------------------------------------------------
# Frame skipping
# -----------------------------------------------------------------------
# MegaDetector runs every FRAME_SKIP frames. SpeciesNet jobs are queued
# after MegaDetector and processed one per frame across the skipped frames,
# spreading the load instead of spiking on a single frame.
FRAME_SKIP = 3

# -----------------------------------------------------------------------
# Load models
# -----------------------------------------------------------------------
print('Loading MegaDetector model (downloads on first run)...')
detector = load_detector('MDV5A')

print('Loading SpeciesNet classifier (downloads on first run)...')
classifier = SpeciesNetClassifier(DEFAULT_MODEL)

print('Models loaded. Starting webcam...')

# -----------------------------------------------------------------------
# Webcam setup
# -----------------------------------------------------------------------
# Capture at 640x480 (~22 FPS) then upscale display to 1280x720.
# MegaDetector uses relative bbox coords (0-1) so detection quality
# is unaffected by capture resolution.
CAPTURE_W, CAPTURE_H = 640, 480
DISPLAY_W, DISPLAY_H = 1280, 720

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
cv2.namedWindow('MegaDetector - Wildlife Detection')

frame_count = 0
last_detections = []   # most recent MegaDetector results
last_species = {}      # maps detection index -> species label string
last_pil_img = None    # PIL image saved from the last detection frame for SpeciesNet

# SpeciesNet job queue - populated after MegaDetector runs, drained one
# item per frame across the skipped frames to spread GPU load evenly
species_queue = deque()

# -----------------------------------------------------------------------
# FPS tracking
# -----------------------------------------------------------------------
# Display FPS: smoothed over last 30 frame timestamps
display_times = deque(maxlen=30)

# Detection FPS: smoothed over last 10 MegaDetector run timestamps
detection_times = deque(maxlen=10)

while True:
    ret, frame = capture.read()
    if not ret:
        sys.exit()

    # Upscale to display resolution — detection runs on the small frame
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    h, w = DISPLAY_H, DISPLAY_W

    # -------------------------------------------------------------------
    # MegaDetector phase - runs every FRAME_SKIP frames
    # -------------------------------------------------------------------
    if frame_count % FRAME_SKIP == 0:

        # MegaDetector expects a PIL Image in RGB (cv2 uses BGR by default)
        last_pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run MegaDetector to find animals/people/vehicles
        result = detector.generate_detections_one_image(
            last_pil_img, detection_threshold=CONFIDENCE_THRESHOLD
        )
        last_detections = result.get('detections', [])
        last_species = {}

        # Queue up SpeciesNet jobs for qualifying animal detections.
        # These will be processed one per frame on subsequent frames
        # rather than all at once here.
        species_queue.clear()
        for i, detection in enumerate(last_detections):
            if detection['category'] == '1' and detection['conf'] >= SPECIES_THRESHOLD:
                species_queue.append(i)

        # Record timestamp for detection FPS smoothing
        detection_times.append(time.time())

    # -------------------------------------------------------------------
    # SpeciesNet phase - process one queued animal per frame
    # -------------------------------------------------------------------
    # By processing one job per frame we spread the classification work
    # across the skipped frames instead of blocking on a detection frame
    if species_queue and last_pil_img is not None:
        i = species_queue.popleft()
        detection = last_detections[i]

        # SpeciesNet bbox uses the same relative format as MegaDetector:
        # [xmin, ymin, width, height] all in 0-1 range
        x, y, bw, bh = detection['bbox']
        bbox = BBox(xmin=x, ymin=y, width=bw, height=bh)

        # Preprocess crops and resizes the animal region for the classifier
        preprocessed = classifier.preprocess(last_pil_img, bboxes=[bbox])

        # predict returns top-5 species - we use the top result
        species_result = classifier.predict('frame', preprocessed)
        classifications = species_result.get('classifications', {})
        classes = classifications.get('classes', [])
        scores = classifications.get('scores', [])

        if classes:
            # SpeciesNet labels are taxonomy strings like "animalia;mammalia;deer"
            # We take the last segment as the readable species name
            top_species = classes[0].split(';')[-1].strip()
            top_score = scores[0] if scores else 0.0
            last_species[i] = f"{top_species}: {top_score:.2f}"

    frame_count += 1

    # -------------------------------------------------------------------
    # Drawing phase - runs every frame using the most recent detections
    # -------------------------------------------------------------------
    for i, detection in enumerate(last_detections):
        conf = detection['conf']
        category = detection['category']

        # Convert relative bbox coords to pixel coords for drawing
        x_rel, y_rel, w_rel, h_rel = detection['bbox']
        x1 = int(x_rel * w)
        y1 = int(y_rel * h)
        x2 = int((x_rel + w_rel) * w)
        y2 = int((y_rel + h_rel) * h)

        color = COLORS.get(category, (255, 255, 255))

        # Use species name if SpeciesNet has finished this detection,
        # otherwise fall back to the generic class label
        if i in last_species:
            label = last_species[i]
        else:
            label = f"{LABEL_MAP.get(category, category)}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------------------------------------------------
    # FPS overlay - top right corner
    # -------------------------------------------------------------------
    now = time.time()
    display_times.append(now)

    # Smooth display FPS: total frames in window / time span of window
    if len(display_times) >= 2:
        display_fps = (len(display_times) - 1) / (display_times[-1] - display_times[0])
    else:
        display_fps = 0.0

    # Smooth detection FPS: same approach over detection timestamps
    if len(detection_times) >= 2:
        detect_fps = (len(detection_times) - 1) / (detection_times[-1] - detection_times[0])
    else:
        detect_fps = 0.0

    disp_text = f"Display: {display_fps:.1f} FPS"
    det_text  = f"Detect:  {detect_fps:.1f} FPS"

    # Right-align both lines - use the wider one to anchor the x position
    size_disp, _ = cv2.getTextSize(disp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    size_det,  _ = cv2.getTextSize(det_text,  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    fps_x = w - max(size_disp[0], size_det[0]) - 10

    cv2.putText(frame, disp_text, (fps_x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, det_text,  (fps_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    cv2.imshow('MegaDetector - Wildlife Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
