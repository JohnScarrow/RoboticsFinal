import cv2
import sys
import numpy as np

def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)
   
def preprocess_img(img):
   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   img = cv2.GaussianBlur(img, (21,21), 0)
   return img
   
capture = cv2.VideoCapture(0)

key = ord('r') # anything other than q

background = None

while True:
   ret, img = capture.read()
   if not ret:
      sys.exit()
   
   gray_img = preprocess_img(img)
   if background is None:
      background=gray_img
      continue
   
   diff = cv2.absdiff(background, gray_img)
   thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
   thresh = cv2.dilate(thresh, None, iterations = 2)
   num_labels, labels = cv2.connectedComponents(thresh.astype(np.uint8))
   
   # loop through the differences and draw boxes
   for label in  range(1, num_labels):
      # current label mask
      mask = (labels == label).astype(np.uint8)
      #get coords
      x, y, w, h = cv2.boundingRect(mask)
      #calculate area
      area = w*h
      # draw bounding box if the area is greater than threshold
      if area>1000:
         cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
   
   show_img(img)

   key = cv2.waitKey(10)
   # set exit condition
   if key== ord('q') or (cv2.getWindowProperty('Image Window', cv2.WND_PROP_VISIBLE) < 1):
      break
   
   
capture.release()
cv2.destroyAllWindows()