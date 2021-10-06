import cv2
import numpy as np
from pyzbar.pyzbar import decode

img = cv2.imread('nokia_barcode.jpg')
#img = cv2.imread('samsung_barcode.jpg')

#cap = cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)

 #success, img = cap.read()
for barcode in decode(img):
    myData = barcode.data.decode('utf-8')
    myType = barcode.type
    if len(myData)==15 and myType=="CODE128":
        print(f"IMEI:{myData},Type:{myType}")
        print(f"Weight of Glass : 10 gms")
        print(f"Weight of Plastic : 8 gms")
        print(f"Weight of Nickel : 6 gms")
        print(f"Total Weights : 24 gms")
        #print(f"Weight of Glass : 12 gms")
        #print(f"Weight of Plastic : 10 gms")
        #print(f"Weight of Nickel : 8 gms")
        #print(f"Total Weights : 30 gms")
    elif len(myData)==13 and myType=="EAN13":
        print(f"SERIAL NUMBER:{myData},Type:{myType}")
    else:
        print(f"ITEM NUMBER:{myData},Type:{myType}") 

    pts = np.array([barcode.polygon],np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(255,0,255),5)
    pts2 = barcode.rect
    cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,
    0.9,(255,0,255),2)

cv2.imshow('Nokia',img)
#cv2.imshow('Samsung',img)
cv2.waitKey(0)