import cv2
import time
import numpy as np
import pickle
from GUI.CAMERA_NUMBERS import *
LINE1_DATA = "C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\GUI\\data\\line1.pickle"
LINE2_DATA = "C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\GUI\\data\\line2.pickle"

class GuiSettings:
    # camera - номер камеры
    def __init__(self, camera=1):
        if camera == 1:
            self.camera = CAMERA1
        else:
            self.camera = CAMERA2
        self.window_name = "CAMERA " + str(self.camera) + " | SETTINGS"
        cv2.namedWindow(self.window_name)
        self.vs = cv2.VideoCapture(self.camera)
        time.sleep(2.0)
        # Началось ли рисование
        self.isStartDraw = False
        # Координаты линии
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

    def start(self):
        self.frame = np.zeros((640,480,3), np.uint8)
        def mouse_listener(event, x, y, flags, param):
            # Смотрим тип события
            # Зажал левую клавишу
            if event == cv2.EVENT_LBUTTONDOWN:
                self.isStartDraw = True
                self.x1, self.y1 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.isStartDraw:
                    self.x2, self.y2 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                self.isStartDraw = False
                self.x2, self.y2 = x, y

        cv2.setMouseCallback(self.window_name, mouse_listener)
        while True:
            fr, self.frame = self.vs.read()
            if not(self.x1 == 0 and self.y1 == 0 and self.x2 == 0 and self.y2 == 0):
                cv2.line(self.frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255)) # RGB наоборот
            cv2.imshow(self.window_name, self.frame)
            key = cv2.waitKey(1) & 0xFF
            # q - закрыть программу
            if key == ord("q"):
                break

        self.vs.release()
        cv2.destroyAllWindows()
        print("[INFO] запись данных о координатах...")
        file = None
        if self.camera == CAMERA1:
            file = open(LINE1_DATA, "wb")
        elif self.camera == CAMERA2:
            file = open(LINE2_DATA, "wb")
        else:
            raise Exception("Select correct number of camera")
        if file is not None:
            file.write(pickle.dumps((self.x1, self.y1, self.x2, self.y2)))
            file.close()
