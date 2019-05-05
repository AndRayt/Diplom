from tkinter import *
from PIL import Image, ImageTk
import cv2
import time
from GUI.CAMERA_NUMBERS import *

from main_2_cams_with_GUI import FrameProc

class GUIStart:
    # line1, line2 - координаты линии входа на 1 и 2 камерах
    def __init__(self, line_1, line_2):
        self.fp = FrameProc(line_1, line_2)
        self.root = Tk()
        self.root.title("Система мониторинга")
        self.frameLeft = Frame(self.root)
        self.frameRight = Frame(self.root)
        self.frameLeft.grid(row=0, column=0)
        self.frameRight.grid(row=0, column=1)
        self.image_label_left = Label(self.frameLeft)
        self.image_label_left.pack()
        self.image_label_right = Label(self.frameRight)
        self.image_label_right.pack()
        self.vs1 = cv2.VideoCapture(CAMERA1)
        time.sleep(2.0)
        self.vs2 = cv2.VideoCapture(CAMERA2)
        time.sleep(2.0)

    def show_vs(self):
        if self.vs1.isOpened() and self.vs2.isOpened():
            ret1, frame1 = self.vs1.read()
            ret2, frame2 = self.vs2.read()
            frame1, frame2 = self.fp.frames_process(frame1, frame2)
            # Меняем BGR на RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame1 = Image.fromarray(frame1)
            imgtk1 = ImageTk.PhotoImage(master=self.frameLeft, image=frame1)
            self.image_label_left.imgtk = imgtk1
            self.image_label_left.configure(image=imgtk1)
            frame2 = Image.fromarray(frame2)
            imgtk2 = ImageTk.PhotoImage(master=self.frameRight, image=frame2)
            # это присвоение нужно чтобы изображение не собрал сборщик мусора!
            self.image_label_right.imgtk2 = imgtk2
            self.image_label_right.configure(image=imgtk2)
        self.image_label_right.after(10, self.show_vs)

    def start(self):
        self.show_vs()
        self.root.mainloop()
        self.vs1.release()
        self.vs2.release()