
# Face Detection using Python

The simplest face recognition library in the world lets you identify and work with faces from Python or the command line.With respect to the Labeled Faces in the the norm, the accuracy of the model is  high.

## Features

- Find faces in pictures ie:-Locate every face that appears in a photo.

<!-- Importing open cv library -->

import cv2

<!-- for different color popup for face -->

from random import randrange as r

<!-- dataset  -->

trainedData=cv2.CascadeClassifier(r'face.xml')

<!-- choose a image -->
img=cv2.imread(r'vinamra.jpg')

<!-- Display  image -->
cv2.imshow('single person',img)

<!-- PAUSE EXECUTION OF THE PROGRAM TILL ANYKEY IS PRESSED -->
cv2.waitKey()

<!-- image conversion into grey scale -->
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('single person',grayimg)
cv2.waitKey()

<!-- detect faces -->
faceCoordinates=trainedData.detectMultiScale(grayimg)
print(faceCoordinates)

for x,y,w,h in faceCoordinates:cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)

<!-- show the image -->
cv2.imshow('Window',img)
cv2.waitKey()

print('end of program')







## Installation

Python 3.3+ or Python 2.7

Then, install the module from pypi using pip3 (for Python)

pip2 install  cv2
    
## Authors

- [Mandavi Shukla](mandavishukla.hnb@gmail.com)

