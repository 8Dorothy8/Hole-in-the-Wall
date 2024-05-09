# Testing the contouring functionalities of the project.
# Code adapted from https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/

import cv2 
import os

def open_image_with_contour(filepath):
    img = cv2.imread(filepath)

    # resize image to standard and to adjust to distortions
    img = cv2.resize(img, (500,500))
    img = cv2.resize(img, None, fx = 0.75, fy = 0.75)
    
    # converting image into grayscale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # setting threshold of gray image 
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    
    # using a findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 


    contours = [contour for contour in contours if cv2.contourArea(contour) < 100000 and cv2.contourArea(contour) > 20000]
    # kick --> 80k frame 17.5k person
    # jumping --> 300k frame 61k person
    # Lsit --> 29k frame 8.2k personq

    cv2.drawContours(img, contours, -1, (0, 0, 255), 5) 
    
    # displaying the image after drawing contours 
    cv2.imshow('shapes', img)
    #cv2.imshow('shapes', contour)
    
# open_image_with_contour('data/medium/man.png')

if __name__=="__main__":
    dir = "data/easy/"
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        open_image_with_contour(f)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 