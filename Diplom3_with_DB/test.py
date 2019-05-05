import cv2
import time

# Разрешение обеих камер 480x640 (3 RGB канала)
vs1 = cv2.VideoCapture(1)
time.sleep(2.0)
vs2 = cv2.VideoCapture(2)
time.sleep(2.0)

while True:
    ret1, frame1 = vs1.read()
    cv2.imshow('CAMERA 1', frame1)
    ret2, frame2 = vs2.read()
    cv2.imshow('CAMERA 2', frame2)
    key = cv2.waitKey(1) & 0xFF
    # q - закрыть программу
    if key == ord("q"):
        break


