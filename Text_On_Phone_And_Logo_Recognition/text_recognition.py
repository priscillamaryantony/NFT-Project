import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re

#img = cv2.imread('samsung_phone.jpg')
#img = cv2.imread('vivo_phone.jpg')
#img = cv2.imread('Samsung_Logo.png')
img = cv2.imread('Vivo_Logo.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

#image = cv2.imread('samsung_phone.jpg')
#image = cv2.imread('vivo_phone.jpg')
#image = cv2.imread('Samsung_Logo.png')
image = cv2.imread('Vivo_Logo.jpg')

gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img) 
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#osd = pytesseract.image_to_osd(img)
#angle = re.search('(?<=Rotate: )\d+', osd).group(0)
#script = re.search('(?<=Script: )\d+', osd).group(0)
#print("angle: ", angle)
#print("script: ", script)

#custom_config = r'--oem 3 --psm 6 outputbase digits'
#print(pytesseract.image_to_string(img, config=custom_config))

custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
print(pytesseract.image_to_string(img, config=custom_config))

#cv2.imshow('Samsung Phone', img)
#cv2.imshow('Vivo Phone', img)
#cv2.imshow('Samsung Logo', img)
cv2.imshow('Vivo Logo', img)
cv2.waitKey(0)