import cv2
import sys
import numpy as np
import time
from collections import deque

def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)

def preprocess_img(img):
   img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   img = cv2.GaussianBlur(img, (21,21), 0)
   return img

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('Image Window')

DELAY = 1.0  # seconds to look back
frame_buffer = deque()  # stores (timestamp, hsv_frame)

while True:
   ret, img = capture.read()
   if not ret:
      sys.exit()

   now = time.time()
   hsv_img = preprocess_img(img)
   frame_buffer.append((now, hsv_img))

   # pop frames older than DELAY, keeping the most recent one that is >= DELAY old
   while len(frame_buffer) > 1 and (now - frame_buffer[1][0]) >= DELAY:
      frame_buffer.popleft()

   # not enough history yet, just show the raw feed
   if (now - frame_buffer[0][0]) < DELAY:
      show_img(img)
      key = cv2.waitKey(10)
      if key == ord('q'):
         break
      continue

   background = frame_buffer[0][1]

   diff = cv2.absdiff(background[:,:,2], hsv_img[:,:,2])
   thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
   thresh = cv2.dilate(thresh, None, iterations=4)
   num_labels, labels = cv2.connectedComponents(thresh)

   for label in range(1, num_labels):
      mask = (labels == label).astype(np.uint8)
      x, y, w, h = cv2.boundingRect(mask)
      if w * h > 5000:
         cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

   show_img(img)

   key = cv2.waitKey(10)
   if key == ord('q'):
      break

capture.release()
cv2.destroyAllWindows()
