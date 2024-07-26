
#Import necessary libraries:
import numpy as np
import imutils
import cv2
from pyzbar import pyzbar
from pyzbar.pyzbar import decode
from PIL import Image

#Load the image:
image = cv2.imread("images/Multiple_barcodes.png") #<-- Multiple_barcodes.png

#Convert the image to grayscale:
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #This line converts the loaded image from the BGR color space to grayscale.
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)

#Calculate the gradient magnitude:
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient) #These lines subtract the Y gradient 
#from the X gradient and convert the result to absolute values.

#This line(s) applies a blur to the gradient image using a various sized kernels.
# (3,7) by itself, it only detects all barcodes but counts 4 with 1 barcode content.
# (8,8) & (3,7), it only detects all barcodes but counts 4 groupings with 1 barcode content.
# **(9,5) & (3,7)**, together detects all barcodes but counts 4 groupings with 1 barcode content.
blurred = cv2.blur(gradient, (9, 5)) 
blurred = cv2.blur(gradient, (3, 7)) 
#blurred = cv2.blur(gradient, (8, 8)) <-- (8,8) by itself can detect *5* seperate barcodes but no decoding. 

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
blurred = cv2.erode(blurred, kernel, iterations = 4)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
blurred = cv2.dilate(blurred, kernel, iterations = 4)

#Threshold the blurred image:
(_, thresh) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
#This line applies a binary threshold to the blurred image, converting it to 
# a binary image where pixels above a certain threshold value (230 in this case) 
# are set to 255 (white) and pixels below the threshold are set to 0 (black).
# **** no box below 230. less box from 234.
#Perform morphological operations:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
closed = cv2.dilate(closed, kernel, iterations = 3)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
closed = cv2.erode(closed, kernel, iterations = 7)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
closed = cv2.dilate(closed, kernel, iterations = 4)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
closed = cv2.erode(closed, kernel, iterations = 3)

#Find contours: This line finds contours in the binary image.
contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#Filter and draw contours:
#These lines filter the contours based on their area and draw the filtered contours on the image.
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

barcode_count = 0  # Counter for barcodes

for i, cnt in enumerate(cnts):
    if cv2.contourArea(cnt) > 2000:
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
        barcode_count += 1
        # Put coloured index number on the rectangle
        cv2.putText(image, str(i+1), (cnt[0][0][0], cnt[0][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Decode the barcode
        x, y, w, h = cv2.boundingRect(cnt)
        barcode_roi = gray[y:y + h, x:x + w]
        barcode = pyzbar.decode(barcode_roi)
        if barcode:
            barcode_data = barcode[0].data.decode("utf-8")
            barcode_type = barcode[0].type
            cv2.putText(image, f"{barcode_data} ({barcode_type})",
                        (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Barcode {i+1} content: {barcode_data}")

print("Number of barcodes found:", barcode_count)

# These lines create a named window, 
# display the image, and close terminal to close the window.
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()