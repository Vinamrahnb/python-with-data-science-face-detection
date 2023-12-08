#->importing open cv library
import cv2
#for different color popup for face
from random import randrange as r
#->dataset lo
trainedData=cv2.CascadeClassifier(r'face.xml')
#->choose a image
img=cv2.imread(r'vinamra.jpg')
#->display  image
cv2.imshow('single person',img)
#->PAUSE EXECUTION OF THE PROGRAM TILL ANYKEY IS PRESSED
cv2.waitKey()

#->image conversion into grey scale
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('single person',grayimg)
cv2.waitKey()

#->detect faces
faceCoordinates=trainedData.detectMultiScale(grayimg)
print(faceCoordinates)

for x,y,w,h in faceCoordinates:cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)

#show the image
cv2.imshow('Window',img)
cv2.waitKey()

print('end of program')