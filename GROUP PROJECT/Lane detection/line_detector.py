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

#set up the area of interest
def region_of_interest(canny):
    
    height = canny.shape[0]
    width =canny.shape[1]
    
    #zero_loke does not change the dimension of an input arrayq
    mask = np.zeros_like(canny)
    triangle = np.array([[(200, height),
                          (800, 350),
                          (1200, height), ]], np.int32)
    #fillPoly masks out everything apart from the triangle
    cv2.fillPoly(mask, triangle, 255)
    #bitwise_and helps to extracts only triangle area
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
#drawing lines on an image
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
    return line_image

#defines which line is imporant and which is not
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

#defining slope of the lines intercept
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def text():
    width  =int(cap.get(3))
    height  =int(cap.get(4))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = cv2.putText(combo_image, 'LINES DETECTOR TESTING at Strive School', (250, height -20),
                      font,1, (255,255,255), 5, cv2.LINE_AA)
    return text


cap = cv2.VideoCapture('test1.mp4')
while True:
    ret, frame = cap.read()
    
    #detecting edges
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    #detecting lines 
    lines= houghLines(cropped_canny)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = addWeighted(frame, line_image)
    text()
   
    
    
    
    cv2.imshow('line_detector', combo_image)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()