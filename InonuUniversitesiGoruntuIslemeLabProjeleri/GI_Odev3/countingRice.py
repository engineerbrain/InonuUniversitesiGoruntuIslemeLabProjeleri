import cv2
import numpy as np

def convert_to_grayscale(image):
    img_gray = np.zeros_like(image[:,:,0])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            R = image[i, j, 2]
            G = image[i, j, 1]
            B = image[i, j, 0]
            gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            img_gray[i, j] = gray
    return img_gray

def find_contours(thresholded):
    contours = []
    
    height, width = thresholded.shape
    
    for i in range(height):
        for j in range(width):
            if thresholded[i, j] == 255:
        
                contour = []
                
                contour.append((j, i))
                
                stack = [(i, j)]
                while stack:
                    current_i, current_j = stack.pop()
   
                    if 0 <= current_i < height and 0 <= current_j < width and thresholded[current_i, current_j] == 255:
         
                        contour.append((current_j, current_i))
      
                        stack.extend([(current_i + di, current_j + dj) for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]])
                
           
                contours.append(contour)
    
    return contours


image = cv2.imread('img.jpg')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Original Image", image)


gray_image = convert_to_grayscale(image)


threshold = 100

thresholded = np.zeros_like(gray_image)


for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        if gray_image[i, j] > threshold:
            thresholded[i, j] = 255


cv2.imshow('Thresholded Image', thresholded)


contours = find_contours(thresholded)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

rice_count = len(contours)
print("Number of Rice:", rice_count)


cv2.imshow('Rice', image)

cv2.waitKey()
