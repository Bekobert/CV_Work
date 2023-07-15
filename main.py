import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera not Working..')
    exit()
    

while True:

    ret, frame = cap.read()

    if not ret:
        print('Error Capturing Camera')
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()