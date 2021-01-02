import glob
import cv2
import os.path

cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('test12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
faces = cascade.detectMultiScale(gray,1.1,3,cv2.CASCADE_SCALE_IMAGE,(24,24))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    face = img[y: y+h, x:x+w, :]
    cv2.imshow('test', img)


 
