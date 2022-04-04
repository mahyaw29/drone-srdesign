import cv2
import numpy as np
import os

# initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# connect to camera
cap = cv2.VideoCapture(0)

pixelWidth=1
width=45
distance=300
imageWidth=0

isf=os.path.isfile('DistanceImage.png')
print(isf)
distanceImage = cv2.imread('DistanceImage.png')
 

# use a grey image
gray = cv2.cvtColor(distanceImage, cv2.COLOR_RGB2GRAY)


# return bounding boxes
boxes, weights = hog.detectMultiScale(distanceImage, winStride=(8,8) )

boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

count=0
for (xLeft, yLeft, xRight, yRight) in boxes:
    # output box on the image
    cv2.rectangle(distanceImage, (xLeft, yLeft), (xRight, yRight),
                      (0, 255, 0), 2)
    cv2.putText(distanceImage, f'P{count}', (xLeft, yLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    count += 1
    print(xLeft,yLeft,xRight,yRight)
    imageWidth=xRight-xLeft

imageWidth=100
focalLength = (imageWidth * distance) / width
#print("ImageWidth: ",imageWidth)
#print("FocalLength: ",focalLength)

while(True):
    #Read in frames from camera
    ret, frame = cap.read()

    # resize image
    frame = cv2.resize(frame, (640, 480))
    # use a grey image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # return bounding boxes
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    count=0
    for (xLeft, yLeft, xRight, yRight) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xLeft, yLeft), (xRight, yRight),
                          (0, 255, 0), 2)
        cv2.putText(frame, f'P{count}', (xLeft, yLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        count += 1
        print(xLeft,yLeft,xRight,yRight)
        pixelWidth=xRight-xLeft
    
    frameDistance = (width * focalLength)/pixelWidth
    if frameDistance==30000:
        frameDistance=0
    #print("PixelWidth: ",pixelWidth)
    if frameDistance!=0:
        print("Distance:",frameDistance)
    #cv2.putText(frame, f'P{count}', (xLeft, yLeft), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyLeftllWindows()
