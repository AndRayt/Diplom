import os
import cv2
import pickle
import numpy as np
from collections import OrderedDict
from preprocessing import alignment_face
TEST_MODE = False

class FaceNetRecognition:
    def __init__(self,
                 recognition_network="C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\recognition\\FaceNet\\openface_nn4.small2.v1.t7",
                 face_classifier="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//dump_data//face_classifier.pickle",
                 label_encoder="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//dump_data//labels.pickle",
                 employees_list = "C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//dump_data//known_employees.pickle"):

        # загрузка модели распознователя лиц
        print("[INFO] загрузка распознователя лиц...")
        self.recognition_network = cv2.dnn.readNetFromTorch(recognition_network)

        # загрузка классификатора с метками
        print("[INFO] загрузка классификатора лиц...")
        self.face_classifier = pickle.loads(open(face_classifier, "rb").read())
        self.label_encoder = pickle.loads(open(label_encoder, "rb").read())

        # загрузка листа известных работников
        employees_list = pickle.loads(open(employees_list, "rb").read())
        # Переводим лист в словарь по ID
        self.employee_dict = OrderedDict()
        for employee in employees_list:
            self.employee_dict[employee.id] = employee

    def human_recognition(self, frame, face_box):
        (startX, startY, endX, endY) = face_box
        # получаем ROI лица
        face = frame[startY:endY, startX:endX]
        # выравниваем лицо по линии глаз
        face = alignment_face(face)
        # создаем блоб и подаек его на классификатор
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.recognition_network.setInput(faceBlob)
        vec = self.recognition_network.forward()

        # получаем наиболее вероятный label (имя) лица
        preds = self.face_classifier.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = self.label_encoder.classes_[j]
        return name
