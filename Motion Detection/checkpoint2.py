import cv2
import sys

# Checkpoint 2 - Live Feed
# CS 280 Introduction to Robotics
# John - North Idaho College

# reusing show_img from checkpoint 1 but without the blocking loop
# since we handle the exit condition in the main loop now
def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)

# 0 means the default webcam
capture = cv2.VideoCapture(0)

while True:
   ret, img = capture.read()
   # ret is false if the webcam disconnects or fails
   if not ret:
      sys.exit()

   # convert each frame to HSV like we did with the static image in checkpoint 1
   # had to make sure to do this every frame not just once
   # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # forgot .copy() at first but turns out its fine here
   hsv_img = img.copy()
   hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

   show_img(hsv_img)

   key = cv2.waitKey(10)
   if key == ord('q'):
      break

capture.release()
cv2.destroyAllWindows()
