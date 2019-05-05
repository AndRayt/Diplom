import cv2
import time
from preprocessing import frame_preprocessing

vs1 = cv2.VideoCapture(0)
time.sleep(2.0)
while True:
    sf, frame = vs1.read()
    frame = frame_preprocessing(frame)
    cv2.imshow('camera_test', frame)
    key = cv2.waitKey(1) & 0xFF
    # q - закрыть программу
    if key == ord("q"):
        break