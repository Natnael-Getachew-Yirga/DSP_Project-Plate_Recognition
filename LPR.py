#press q to see each phase implementation and quite
import cv2
import imutils
import numpy as np
import pytesseract

img = cv2.imread('3.jpg', cv2.IMREAD_COLOR)  # import our image
cv2.imshow("Original Image", img)       # display imported image
cv2.waitKey(0)
img = cv2.resize(img, (620, 480))       # resize the image
cv2.imshow("Resized Image", img)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
gray = cv2.bilateralFilter(gray, 11, 15,15)  # Blur to reduce noise
cv2.imshow("FIlter Image", gray)
cv2.waitKey(0)
edged = cv2.Canny(gray, 10,200)  # Perform Edge detection
cv2.imshow("Canny eaged Image", edged)
cv2.waitKey(0)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours= sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None

for contour in contours:
    # approximate the contour
    approx = cv2.approxPolyDP(contour, 10, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("only plate Image", new_image)
cv2.waitKey(0)
#for croping (character segmentation)
(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
cv2.imshow("cropped image", cropped_image)
cv2.waitKey(0)


#Read the number plate
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
plate = pytesseract.image_to_string(cropped_image, config='')
print("Detected Number is:",plate)
cv2.imshow('image',img)
cv2.imshow('Cropped',cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


