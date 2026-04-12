import cv2
import numpy as np

# Checkpoint 1 - Static Image Processing
# CS 280 Introduction to Robotics
# John - North Idaho College

# show_img displays the image and waits for q to quit
def show_img(img_to_show):
   cv2.imshow('Image Window', img_to_show)

   key = ord('r')
   while key != ord('q'):
      key = cv2.waitKey(10)
   cv2.destroyAllWindows()

# read in the jellyfish image
img = cv2.imread('jellyfish.jpg')

# print the shape so I can see (height, width, channels)
print(img.shape)

# show the original image first
show_img(img)

# --- Grayscale ---
# I tried doing this without .copy() first and it messed up the original
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = img.copy()
gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
# shape should only have 2 values now since grayscale is 1 channel
print(gray_img.shape)
show_img(gray_img)

# --- HSV ---
# HSV separates color info from brightness which is useful for
# processing later since lighting changes wont throw everything off
hsv_img = img.copy()
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)
print(hsv_img.shape)
show_img(hsv_img)
