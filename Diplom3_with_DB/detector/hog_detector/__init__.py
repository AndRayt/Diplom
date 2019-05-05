import dlib
import cv2

class HOGDetection:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def _frame_preprocessing(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def human_detection(self, frame):
        boxes = self.detector(frame, 0)
        result = []
        for box in boxes:
            x1 = box.left()
            y1 = box.top()
            x2 = box.right()
            y2 = box.bottom()
            result.append((x1, y1, x2, y2))
        return result