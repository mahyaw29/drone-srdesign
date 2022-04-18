import cv2
import numpy as np
import os
from imutils.object_detection import non_max_suppression


# initialize HOG descriptor
hogDetect = cv2.HOGDescriptor()
hogDetect.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# connect to camera
capture = cv2.VideoCapture(0)

#For video capture
output = cv2.VideoWriter('hd.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (320,240))


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
b, w = hogDetect.detectMultiScale(distanceImage, winStride=(8,8) )

b = np.array([[x, y, x + w, y + h] for (x, y, w, h) in b])

count=0
for (xLeft, yLeft, xRight, yRight) in b:
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
    ret, frame = capture.read()

    # resize frame
    frame = cv2.resize(frame, (320, 240))
    # use a grey image
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # return bounding boxes
    b, w = hogDetect.detectMultiScale(frame, winStride=(2,2),padding=(8, 8), scale=1.5 )

    b = np.array([[x, y, x + w, y + h] for (x, y, w, h) in b])
    p= non_max_suppression(b, probs=None, overlapThresh=0.65)

    count=0
    for (xLeft, yLeft, xRight, yRight) in b:
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
    
    #For video capture
    output.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#For video capture
output.release()

capture.release()
cv2.destroyAllWindows()
