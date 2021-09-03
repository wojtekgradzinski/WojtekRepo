import cv2
import numpy as np 


image = cv2.imread('img/Green_lion.jpg')
background = cv2.imread('img/night1.jpg')

#background = np.ones(image.shape,dtype=np.uint8)*144

#changing the background size equal to the image
background = cv2.resize(background, (image.shape[1],image.shape[0]), interpolation = cv2.INTER_AREA)
background = cv2.cvtColor(background,cv2.COLOR_BGR2HSV)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([36,0,0])
upper_green = np.array([86, 255,255])

#mask for the background
mask = cv2.inRange(hsv, lower_green, upper_green)

mask_inv = cv2.bitwise_not(mask)

background = cv2.bitwise_and(background,background, mask=mask)
target = cv2.bitwise_and(hsv,hsv, mask=mask_inv)
target = cv2.bitwise_or(target,background)
print(mask.shape)
BGR = cv2.cvtColor(target,cv2.COLOR_HSV2BGR)
cv2.imshow('lion',BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()