import cv2 
import numpy as np 


def canny(img):
    
    #convert to grayscale for better accuracy    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    #set the kernel size for blur transformation to reduce noise
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    #defining edges with canny
    canny = cv2.Canny(gray, 50,150)
    return canny    













cap = cv2.VideoCapture('test1.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    #detecting edges
    canny_image = canny(frame)
    cv2.imshow('canny_image')