import cv2
import numpy as np

def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)
   
   # set exit condition
   key = ord('r') # anything other than q
   while key != ord('q'):
      key = cv2.waitKey(10)
   cv2.destroyAllWindows()
   
# read in image
img = cv2.imread('jellyfish.jpg')
# show_img(img)

print(img.shape)

# Info about pixel
print(img[67, 76, 2])  # BGR

# Grayscsale
gray_img = img.copy()
gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
show_img(gray_img)
# Info about pixel
print(img[67, 76])

# HSV
hsv_img = img.copy()
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)
print(hsv_img.shape)
show_img(hsv_img)
# Info about pixel
print(img[67, 76, 2])  # HSV
