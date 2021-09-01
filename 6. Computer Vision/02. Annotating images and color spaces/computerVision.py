import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def imshow(img, grayscale = False):
    plt.figure(figsize = (20,15));
    if grayscale:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:,:,::-1])    

path = 'img/purple-flowers.jpg'


image = cv2.imread(path)

cv2.imshow("Hello",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


















low_orange = np.array([95,0,0])
high_orange = np.array([130,255,255])

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    cv2.imshow('Original frame',frame)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,low_orange,high_orange)
    cv2.imshow('Masked frame',mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()