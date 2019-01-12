import numpy as np
import cv2
cap = cv2.VideoCapture(0)
i = 0
max_shots = 6
while(cap.isOpened()):
    ret, frame = cap.read()
    if (ret == True):
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            cv2.imwrite('res'+str(i)+'.png',frame)
            i += 1
        if i == max_shots:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
