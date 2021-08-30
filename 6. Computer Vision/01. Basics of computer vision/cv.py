import cv2
import numpy as np

img =  cv2.imread('01. Basics of computer vision\annotations.jpg')
#img.shape

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)