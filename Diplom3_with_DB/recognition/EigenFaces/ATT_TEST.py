import cv2
import numpy as np
import os
from PIL import Image
from imutils import paths
import time

class EigenFacesRecognition:
    def __init__(self):
        cascadePath = "C:\\Users\\Andrew\\PycharmProjects\\Diplom\\detector\\haar_cascades_detection\\haarcascades\\haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        # 23 - сколько главных компонент мы берем (чем больше тем больше информации)
        # 2500 - порог, если точность классификации ниже то считаем что лицо неизвестно
        self.recog = cv2.face.EigenFaceRecognizer_create(23, 2500.0)

    def learning(self, path, number_of_people=5):
        image_paths = list(paths.list_images(path))
        images = []
        names = []
        start_time_learn = time.time()
        self.x_face_size = 92
        self.y_face_size = 112
        people_count = 0  # счетчик сколько фотографий данного человека уже взято
        last_name = -1
        for image_path in image_paths:
            gray = Image.open(image_path).convert('L')
            image = np.array(gray, 'uint8')
            image = cv2.resize(image, (self.x_face_size, self.y_face_size))
            name = int(image_path.split(os.path.sep)[-2][1])
            if name == last_name:
                people_count += 1
            else:
                people_count = 0
                last_name = name
            if people_count < number_of_people:
                images.append(image)
                names.append(name)
        self.labels = names
        print(self.labels)
        self.recog.train(images, np.array(self.labels))
        self.train_time = time.time() - start_time_learn

    def human_recognition(self, path):
        start_time = time.time()
        image_paths = list(paths.list_images(path))
        self.labels_recog = []
        for image_path in image_paths:
            gray = Image.open(image_path).convert('L')
            image = np.array(gray, 'uint8')
            label, conf = self.recog.predict(image)
            if label == -1:
                self.labels_recog.append(0)
            else:
                self.labels_recog.append(label)
        print(self.labels_recog)
        self.time = time.time() - start_time

    def statistic(self):
        real_labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]
        false_count = 0 # количество неверных ответов
        # смотрим ссответсвие известным ответам
        for i in range(0, len(real_labels)):
            if not (real_labels[i] == self.labels_recog[i]):
                false_count += 1
        # остальные должны быть 0
        for i in range(len(real_labels), len(self.labels_recog)):
            if not (self.labels_recog[i] == 0):
                false_count += 1
        return self.train_time, self.time, false_count

faceRecog = EigenFacesRecognition()
faceRecog.learning("C:\\Users\\Andrew\\PycharmProjects\\Diplom2\\recognition\\ATT_DB\\train", number_of_people=5)
faceRecog.human_recognition("C:\\Users\\Andrew\\PycharmProjects\\Diplom2\\recognition\\ATT_DB\\test")
train_time, time, false_count = faceRecog.statistic()
print("train time:", train_time, "test time:", time, "false count:", false_count)
