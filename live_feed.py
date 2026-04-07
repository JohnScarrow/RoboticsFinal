import cv2
import sys

def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)
   
capture = cv2.VideoCapture(0)

key = ord('r') # anything other than q
while True:
   ret, img = capture.read()
   if not ret:
      sys.exit()
   # HSV
   hsv_img = img.copy()
   hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)   
   show_img(hsv_img)

   key = cv2.waitKey(10)
   # set exit condition
   if key== ord('q') or (cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1):
      break
   
   
capture.release()
cv2.destroyAllWindows()