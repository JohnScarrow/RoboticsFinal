import cv2
import sys
import numpy as np
import time

# Lab 6 - OpenCV
# CS 280 Introduction to Robotics
# John - North Idaho College

# Helper function to show an image in a window
def show_img(img_to_show):
    cv2.imshow('Image Window', img_to_show)

# Preprocess: convert to HSV and blur to reduce noise.
# HSV is better for motion detection than grayscale because it separates
# color (hue) from brightness (value), so it can catch movement even
# when the lighting doesnt change much between frames.
def preprocess_img(img):
    # thought opencv used RGB like everything else but it actually stores as BGR
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # GaussianBlur smooths out small differences so we dont get
    # a ton of false motion detections from like lighting changes
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return img

# --- Live Feed with HSV Preprocessing ---
# I learned from the live_feed example that we can capture frames
# in a loop using VideoCapture. Here I'm preprocessing each frame
# before displaying it, similar to what the motion detector does.

capture = cv2.VideoCapture(0)

background = None
background_time = None  # track when background was last set
saved = False  # only save one preprocessed image as the deliverable

while True:
    ret, img = capture.read()
    if not ret:
        sys.exit()

    hsv = preprocess_img(img)

    # Save one preprocessed frame to disk for the deliverable
    if not saved:
        cv2.imwrite('preprocessed_frame.png', hsv)
        print("Saved preprocessed_frame.png")
        saved = True

    # On the first frame, store it as the background reference.
    # After that, refresh every 5 seconds so lighting changes don't
    # mess up the motion detection over time.
    if background is None or (time.time() - background_time) >= 10:
        background = hsv
        background_time = time.time()
        continue

    # Compute difference between current frame and background
    # tried just subtracting but it underflows and gives weird results for pixels
    # where background is brighter than the current frame
    # diff = background - gray
    diff = cv2.absdiff(background, hsv)

    # HSV gives us 3 channels so we take the max difference across H, S, and V.
    # This way if motion shows up in any channel it gets detected.
    diff = np.max(diff, axis=2)

    # Threshold: anything above 25 brightness difference counts as motion
    thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]

    # Dilate to fill in gaps between nearby motion regions
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find connected components (groups of changed pixels)
    num_labels, labels = cv2.connectedComponents(thresh.astype(np.uint8))

    # Draw a bounding box around anything big enough to be real motion
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        area = w * h
        if area > 2000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    show_img(img)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
