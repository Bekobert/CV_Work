import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import pygetwindow as gw

cap = cv2.VideoCapture(0)

width, height = pyautogui.size()
#s_width, s_height = gw.Size()

pyautogui.FAILSAFE = False



if not cap.isOpened():
    print('Camera not Working..')
    exit()
    
l_skin = np.array([0, 20, 70]).astype(np.uint8)
u_skin = np.array([20, 255, 2550]).astype(np.uint8)

mp_hands = mp.solutions.hands.Hands()
g_result = None

gesture_history = []
gesture_threshold = 30

while True:

    ret, frame = cap.read()

    if not ret:
        print('Error Capturing Camera')
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(rgb_frame)
    
    frame = cv2.flip(frame, 1)
    if g_result != results.multi_hand_landmarks: 
        print(g_result)
        g_result = results
        

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            #print(hand_landmarks)
            joints = []
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            mid_finger_pos = hand_landmarks.landmark[8]
            
            
            mid_finger_x, mid_finger_y = int(mid_finger_pos.x * width), int(mid_finger_pos.y * height)

            movement_scale = 1
            #pyautogui.moveTo(mid_finger_x * movement_scale, mid_finger_y * movement_scale)
            app_window = gw.getActiveWindow()
        

            for landmark in hand_landmarks.landmark:
                joint_x, joint_y, joint_z = landmark.x, landmark.y, landmark.z
                joints.append((joint_x, joint_y, joint_z))

            
            

            fingers_up = sum(joint[1] < joints[7][1] for joint in joints[5:8]) <= 2
            print(joints[8][1])
            if fingers_up:
                gesture = 'Open hand'
            else:
                gesture = 'Closed hand'

            if gesture == 'Closed hand':
                app_window.moveTo(width-(mid_finger_x * movement_scale), mid_finger_y * movement_scale)

            gesture_history.append(gesture)

            

            cv2.putText(frame, gesture, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in range(0, 21):
                cv2.putText(frame, 'joint({finger})'.format(finger = i), (int(joints[i][0] * 600), int(joints[i][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 1)

            #cv2.putText(frame, 'joint({finger})'.format(finger = '8'), (int(joints[8][0] * 600), int(joints[8][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #cv2.putText(frame, 'joint({finger})'.format(finger = '7'), (int(joints[7][0] * 600), int(joints[7][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #cv2.putText(frame, 'joint({finger})'.format(finger = '6'), (int(joints[6][0] * 600), int(joints[6][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #cv2.putText(frame, 'joint({finger})'.format(finger = '5'), (int(joints[5][0] * 600), int(joints[5][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #cv2.putText(frame, 'joint({finger})'.format(finger = '4'), (int(joints[4][0] * 600), int(joints[4][1] * 500)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask = cv2.inRange(hsv, l_skin, u_skin)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(frame, contours, -1, (0, 256, 0), 2)

    

    cv2.imshow('Camera', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()