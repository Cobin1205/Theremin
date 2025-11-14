import numpy as np
import time
import cv2 as cv
import mediapipe as mp
from pyo import *

mpHolistic = mp.solutions.holistic
holisticModel = mpHolistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

#Initialize Pyo
try:
    s = Server(audio='portaudio', duplex=0).boot()
    s.start()
    a = Sine(freq=1000, mul=0.5).out()
except:
    print("Could not initialize pyo")
    exit()


mpDrawing = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)

rightIndexHeight = 1
leftIndexHeight = 0.5

while cap.isOpened():

    #Convert frame to RGB
    ret, frame = cap.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #Get results
    image.flags.writeable = False
    results = holisticModel.process(image)
    image.flags.writeable = True

    #Convert image back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    #Right Hand Landmarks
    mpDrawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mpHolistic.HAND_CONNECTIONS
    )

    #Left Hand Landmarks
    mpDrawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mpHolistic.HAND_CONNECTIONS
    )

    #Control frequency from tip of right index finger
    if results.right_hand_landmarks:
        rightIndexHeight = max(0, 1 - results.right_hand_landmarks.landmark[8].y)
    if results.left_hand_landmarks:
        leftIndexHeight = max(0, 1 - results.left_hand_landmarks.landmark[8].y)

    a.setFreq(1000*rightIndexHeight)
    a.setMul(leftIndexHeight)

    #Flip image so hands match up
    image = cv.flip(image, 1)

    #Show the frame
    cv.putText(img=image, 
               text=str(rightIndexHeight), 
               org=(50, 100), 
               fontFace=cv.FONT_HERSHEY_SIMPLEX, 
               fontScale=0.5, 
               color=(0, 255, 0),
               thickness=2,
               lineType=cv.LINE_AA
               )
    
    cv.imshow("Hand Landmarks", image)

    #Quit program when q is hit
    if cv.waitKey(1) == ord('q'):
        a.stop()
        s.stop()
        break

cap.release()
cv.destroyAllWindows()