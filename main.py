import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera not Working..')
    exit()
    
l_skin = np.array([0, 20, 70], dtype=np.uint8)
u_skin = np.array([20, 255, 2550], dtype=np.uint8)

mp_hands = mp.solutions.hands.Hands()

while True:

    ret, frame = cap.read()

    if not ret:
        print('Error Capturing Camera')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask = cv2.inRange(hsv, l_skin, u_skin)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame, contours, -1, (0, 256, 0), 2)

    frame = cv2.flip(frame, 1)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()