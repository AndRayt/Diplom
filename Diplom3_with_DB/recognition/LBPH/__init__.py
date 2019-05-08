import cv2
import numpy as np
import os
from imutils import paths
from collections import OrderedDict
import pickle
from detector.haar_cascades_detection import *
from entity.employee import *
from database import DataBase
from preprocessing import frame_preprocessing, alignment_face

TEST_MODE = False

# ДЕТЕКТОРЫ для обучения
HAAR_CASCADES = 0
FACE_DETECTION = 1
NN_DETECTION = 2

class LBPHRecognition:
    # names - список имен людей
    # x_face_size, y_face_size - размеры изображения
    def __init__(self, x_face_size=120, y_face_size=120):
        self.db = DataBase()
        # радиус, количество соседей, размер сетки по х и y (в оригинальной статье 8), порог - максимальное расстояние до ближайшего соседа
        self.recog = cv2.face.LBPHFaceRecognizer_create(1, 10, 8, 8, 123)
        self.x_face_size = x_face_size
        self.y_face_size = y_face_size
        # Формируем словари names_labels и labels_names:
        self.id = [employee[0] for employee in self.db.get_employee_lst()]
        self.id_labels = OrderedDict()
        self.labels_id = OrderedDict()
        for counter, name in enumerate(self.id):
            self.id_labels[name] = counter
        for counter, name in enumerate(self.id):
            self.labels_id[counter] = name

    def traning(self, detector_type=HAAR_CASCADES,
                dataset="C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\dataset",
                file_path="C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\recognition\\LBPH\\dump_data\\face_classifier.xml",
                employee_file_path="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//employee_data//known_employees.pickle"):
        image_paths = list(paths.list_images(dataset))
        images = []
        labels = []
        detector = None
        if detector_type == HAAR_CASCADES:
            detector = HaarCascadesDetection()
        else:
            raise Exception("Select correct type of detector!")
        employees_lst = self.db.get_employee_lst()
        id_faces_lst = []  # лист с кортежами (id, faces_dir)
        for employee in employees_lst:
            id_faces_lst.append((employee[0], employee[3]))
        for employee in id_faces_lst:
            id = employee[0]
            for image_path in list(paths.list_images(employee[1])):
                image = cv2.imread(image_path)
                image = frame_preprocessing(image)
                label = self.id_labels.get(id)
                faces = detector.human_detection(image)
                for index_face, (x, y, x2, y2) in enumerate(faces):
                    image = image[y: y2, x: x2]
                    if not(len(image) == 0):
                        image = cv2.resize(image, (self.x_face_size, self.y_face_size))
                        if TEST_MODE:
                            cv2.imshow('image train', image)
                            cv2.waitKey(0)
                        images.append(image)
                        labels.append(label)
                        break
        self.recog.train(images, np.array(labels))
        # Сохраняем обученный список сотрудников и распознаватель лиц
        print("[INFO] сохранение базы данных людей...")
        employees = []
        for name in self.id:
            employee = Employee(name)
            employees.append(employee)
        file = open(employee_file_path, "wb")
        file.write(pickle.dumps(employees))
        file.close()
        print("[INFO] сохранение обученной модели распознователя лиц LBPH...")
        self.recog.save(file_path)

    def human_recognition(self, frame, face_box,
                          face_recog_path="C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\recognition\\LBPH\\dump_data\\face_classifier.xml"):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (startX, startY, endX, endY) = face_box
        # получаем ROI лица
        face = frame[startY:endY, startX:endX]
        # выравниваем лицо по линии глаз
        face = alignment_face(face)
        self.recog.read(face_recog_path)
        face = cv2.resize(face, (self.x_face_size, self.y_face_size))
        if TEST_MODE: cv2.imshow('face', face)
        label, conf = self.recog.predict(face)
        if label == -1:
            name = 0 # Возвращаем ID неизвестного
        else:
            name = self.labels_id[label]
        return name

if __name__ == '__main__':
    rec = LBPHRecognition()
    rec.traning()