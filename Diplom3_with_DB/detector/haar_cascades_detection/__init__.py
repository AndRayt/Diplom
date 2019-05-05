import cv2

class HaarCascadesDetection:
    # MODE: РЕЖИМЫ РАБОТЫ ДЕТЕКТОРА
    FULL_BODY = 0
    UPPER_BODY = 1
    FACE = 2
    def __init__(self, mode=FACE):
        self.cascade = None
        if mode == self.FULL_BODY:
            self.cascade = cv2.CascadeClassifier(
                "C:\\Users\\Andrew\\PycharmProjects\\Diplom\\detector\\haar_cascades_detection\\haarcascades\\haarcascade_fullbody.xml")
        elif mode == self.FACE:
            self.cascade = cv2.CascadeClassifier(
                "C:\\Users\\Andrew\\PycharmProjects\\Diplom\\detector\\haar_cascades_detection\\haarcascades\\haarcascade_frontalface_default.xml")
        elif mode == self.UPPER_BODY:
            self.cascade = cv2.CascadeClassifier(
                "C:\\Users\\Andrew\\PycharmProjects\\Diplom\\detector\\haar_cascades_detection\\haarcascades\\haarcascade_upperbody.xml")

    def human_detection(self, frame):
        #1.3, 5 - face
        boxes = self.cascade.detectMultiScale(frame, 1.3, 5)
        result = []
        for (x, y, w, h) in boxes:
            startX, startY, endX, endY = x, y, x + w, y + h
            result.append((startX, startY, endX, endY))
        return result
