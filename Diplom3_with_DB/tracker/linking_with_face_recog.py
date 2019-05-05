from recognition.FaceNet import FaceNetRecognition
from recognition.EigenFaces import EigenFacesRecognition
from recognition.FisherFaces import FisherFacesRecognition
from recognition.LBPH import LBPHRecognition

from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as sp_dist

"""
    Класс подсчитывает лица, сравнивает с прошлым кадром и вызывает разпознование если необходимо
"""

TEST_MODE = False

# Способ распознования людей
FACENET = 0
EIGENFACES = 1
FISHERFACES = 2
LBPH = 3

class SimpleLinkingWithRecog:
    # maxDisappeared - Максимальное количество кадров на котором не видно человека
    # maxDistance - Максимальное расстояние для связи центроид
    # maxFaceK - коэфицент определяющий насколько далеко могут быть
    # центройды лица и тела
    def __init__(self, maxDisappeared=200, maxOldNewDist=100, recog_method=FACENET):
        # Распознователь лица
        if recog_method == FACENET:
            self.rec = FaceNetRecognition()
        elif recog_method == EIGENFACES:
            self.rec = EigenFacesRecognition()
        elif recog_method == FISHERFACES:
            self.rec = FisherFacesRecognition()
        elif recog_method == LBPH:
            self.rec = LBPHRecognition()
        else:
            raise Exception("Select correct recognition method")
        # ID Центроид лица
        self.face_centroids = OrderedDict()
        # Количество кадров когда объект потерян
        self.disappeared = OrderedDict()
        # Максимально допустимое расстояние между лицами на двух кадрах
        self.maxOldNewDist = maxOldNewDist
        self.maxDisappeared = maxDisappeared

    def register(self, face_centroid, face_box, frame):
        name = self.rec.human_recognition(frame, face_box)
        if TEST_MODE: print("Регистрация", self.frame_counter, name)
        # Смотрим был ли такой человек уже найден
        # и если да то удаляем информацию о нем
        self.deregister(name)
        self.face_centroids[name] = face_centroid
        self.disappeared[name] = 0

    def deregister(self, name):
        # Проверяем есть ли такой ключ
        found = False
        for key in self.face_centroids.keys():
            if key == name:
                found = True
                break
        if found:
            if TEST_MODE: print("Удалили:", self.frame_counter, name)
            del self.face_centroids[name]
            del self.disappeared[name]

    # Возвращает массив центроид (для входа - массива)
    def get_centroid_list(self, input_list):
        # Инициализируем массив для текущих центроид (два столбца - две координаты)
        centroids = np.zeros((len(input_list), 2), dtype="int")
        # Проходимся по всем боксам
        for (i, (startX, startY, endX, endY)) in enumerate(input_list):
            # Определяем центр боксов
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            # Сохраняем их в массиве центроид
            centroids[i] = (cX, cY)
        return centroids

    # Словарь в numpy array
    def dict_to_lst(self, dict):
        dict_size = 0
        for key, value in dict.items():
            dict_size += 1
        result = np.zeros((dict_size, 2), dtype="int")
        count = 0
        for key, value in dict.items():
            result[count] = value
            count += 1
        return result

    frame_counter = 0
    # faces_boxes - боксы лица
    def update(self, face_boxes, frame):
        if TEST_MODE: print(face_boxes)
        face_centroids_new_list = self.get_centroid_list(face_boxes)
        # массив уже используемых индексов
        DIST = sp_dist.cdist(self.dict_to_lst(self.face_centroids), face_centroids_new_list)
        if DIST.size != 0:
            # Получаем списки имен по индексам
            row_names = list(self.face_centroids.keys())
            # cols - индексы старого списка лиц
            # rows - индексы нового списка лиц
            used_rows = set()
            used_cols = set()
            count_iteration = 0
            if DIST.shape[0] > DIST.shape[1]:
                count_iteration = DIST.shape[1]
            else:
                count_iteration = DIST.shape[0]
            for i in range(count_iteration):
                min = self.maxOldNewDist
                row_min, col_min = 0, 0
                for row in range(DIST.shape[0]):
                    if row not in used_rows:
                        for col in range(DIST.shape[1]):
                            if col not in used_cols:
                                if DIST[row, col] < min:
                                    min = DIST[row, col]
                                    row_min, col_min = row, col
                self.face_centroids[row_names[row_min]] = face_centroids_new_list[col_min]
                used_rows.add(row_min)
                used_cols.add(col_min)
            # Смотрим лица которым не нашли соответсвие в новом кадре
            for col in range(DIST.shape[0]):
                if col not in used_rows:
                    self.disappeared[row_names[col]] += 1
                    if self.disappeared[row_names[col]] >= self.maxDisappeared:
                        self.deregister(row_names[col])
            # Смотрим новые в кадре лица
            for col in range(DIST.shape[1]):
                if col not in used_cols:
                    self.register(face_centroids_new_list[col], face_boxes[col], frame)
        else:
            # Если до этого лиц не было то регестрируем все лица
            if not(len(face_centroids_new_list) == 0):
                for index, face_centroid in enumerate(face_centroids_new_list):
                    self.register(face_centroid, face_boxes[index], frame)
            else:
                # Иначе значит все зарегестрированные лица не найденны
                for key in self.face_centroids.keys():
                    self.disappeared[key] += 1
                    if self.disappeared[key] >= self.maxDisappeared:
                        self.deregister(key)
        self.frame_counter += 1
        return self.face_centroids


