import numpy as np
import time
import cv2 as cv
import mediapipe as mp

mpHolistic = mp.solutions.holistic
holisticModel = mpHolistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

mpDrawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    #frame = cv.resize(frame, (600, 800))
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holisticModel.process(image)
    image.flags.writeable = True

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

    image = cv.flip(image, 1)

    cv.imshow("Facial Landmarks", image)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()