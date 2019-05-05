from preprocessing import frame_preprocessing
from tracker.linking_with_2_face_recog import *
from entity.employee import Employee
from detector.haar_cascades_detection import HaarCascadesDetection
from database import DataBase
from detector.hog_detector import HOGDetection
from collections import OrderedDict
from imutils.video import FPS
import numpy as np
import time
import dlib
import cv2
import pickle

# 640 x 480 - РАЗМЕР КАДРА

TEST_MODE = False # Вывод тестовой информации
SKIP_FRAMES = 30 # Количество пропускаемых между детектированиями кадров
RECOGNITION_METHOD = FACENET # Метод распознования лиц

class FrameProc:
    def __init__(self, line_1, line_2):
        print("[INFO] начинаем видеопоток...")

        # Линии
        self.line_1 = line_1
        self.line_2 = line_2

        # База данных сотрудников
        self.db = DataBase()

        # переменные для высоты и ширины кадра
        self.W = None
        self.H = None

        # инициализируем трекер центроид
        self.linking_tracker = SimpleLinkingWithRecog(recog_method=RECOGNITION_METHOD)
        # Инициализируем детектор
        self.detector = HaarCascadesDetection(mode=HaarCascadesDetection.FACE)
        #self.detector = HOGDetection()
        self.trackers1 = []
        self.trackers2 = []
        # Словарь вида objectID : TrackableOjbect
        self.trackableObjects = {}

        # Количество обработанных кадров
        self.totalFrames = 0
        # Подсчет людей внутри предприятия
        self.totalPeopleCount = 0
        # Параметры бокса двери
        self.door_points = ((260, 10), (380, 170))
        # Получаем список известных работников
        print("[INFO] загрузка зарегестрирвоанных работников в системе...")
        self.employees_list = pickle.loads(open("C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//employee_data//known_employees.pickle", "rb").read())
        # Переводим лист в словарь по ID
        self.employee_dict = OrderedDict()
        for employee in self.employees_list:
            self.employee_dict[employee.id] = employee

    # Обрабатываем кадры
    def frames_process(self, frame1, frame2):
        np.seterr(over='raise')
        # Предобработка кадра для dlib
        gray_frame1 = frame_preprocessing(frame1)
        gray_frame2 = frame_preprocessing(frame2)
        # Устанавливаем H W
        if self.W is None or self.H is None:
            (self.H, self.W) = frame1.shape[:2]

        # Инициализируем статутс "Ожидание"
        # Статусы:
        # Ожидания: В этом состоянии мы ждем на людей, чтобы быть обнаружены и отслеживаются.
        # Обнаружение: Мы активно в процессе обнаружения людей, используя MobileNet SSD.
        # Отслеживания: Люди отслеживаются в кадре, и мы рассчитываем totalUp и totalDown
        status = "Waiting"

        # Прямоугольники
        # Первая камера
        rects1 = []
        # Вторая камера
        rects2 = []
        # Пропускаем некоторое количество кадров (чтобы улучшить производительность)
        #if self.totalFrames % SKIP_FRAMES == 20:
        trackers1, trackers2 = [], []
        # Начинаем обнаружение
        status = "Detecting"
        faceTrackers = []
        # Получаем массив боксов людей в кадре 1
        list_human_box1 = self.detector.human_detection(frame1)
        # Получаем массив боксов людей в кадре 2
        list_human_box2 = self.detector.human_detection(frame2)
        # Перебираем боксы людей на первом кадре
        for human_box in list_human_box1:
            (startX, startY, endX, endY) = human_box
            # строим прямоугольник dlib и начинаем слежение за ним
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(gray_frame1, rect)
            # добавляем трекер данного кадра в массив трекеров
            trackers1.append(tracker)
            # добавляем эту информацию в массив прямоугольников
            rects1.append((startX, startY, endX, endY))
        # Перебираем боксы людей на втором кадре
        for human_box in list_human_box2:
            (startX, startY, endX, endY) = human_box
            # строим прямоугольник dlib и начинаем слежение за ним
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(gray_frame2, rect)
            # добавляем трекер данного кадра в массив трекеров
            trackers2.append(tracker)
            # добавляем эту информацию в массив прямоугольников
            rects2.append((startX, startY, endX, endY))
        # Не анализируя людей в кадре анализируем их перемещение с помощью трекера
        """"
        else:
            # проходим через все трекеры
            for tracker in self.trackers1:
                # обновляем статус на отслеживание
                status = "Tracking"
                # получаем позицию объекта
                tracker.update(gray_frame1)
                pos = tracker.get_position()
                # получаем координаты позиции
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # добавляем эту информацию в массив прямоугольников
                rects1.append((startX, startY, endX, endY))

            for tracker in self.trackers2:
                # обновляем статус на отслеживание
                status = "Tracking"
                # получаем позицию объекта
                tracker.update(gray_frame2)
                pos = tracker.get_position()
                # получаем координаты позиции
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                # добавляем эту информацию в массив прямоугольников
                rects2.append((startX, startY, endX, endY))
        """

        # Линия относительно которой считается что люди идут вверх или вниз
        cv2.line(frame1, (self.line_1[0], self.line_1[1]), (self.line_1[2], self.line_1[3]), (0, 0, 255))
        cv2.line(frame2, (self.line_2[0], self.line_2[1]), (self.line_2[2], self.line_2[3]), (0, 0, 255))

        # связываем старые центроиды с новыми вычисленными
        objects1, objects2 = self.linking_tracker.update(rects1, frame1, rects2, frame2)
        if TEST_MODE: print("obj1:", objects1, "obj2", objects2)

        # определяем человек пошел вверх или вниз
        # перебираем людей на 1 кадре
        for (objectID, centroid) in objects1.items():
            # получаем объект по ID
            to = self.trackableObjects.get(objectID)

            # если этот объект не существует то создаем его
            if to is None:
                to = self.employee_dict[objectID]
                to.centroids1.append(centroid)
            # иначе определяем движется он вверх или вниз
            else:
                # Разница между текущей y координатой и средней центроиды
                # покажет направление (отрицательный - вверх; положительный - вниз)
                y = [c[1] for c in to.centroids1]
                direction = centroid[1] - np.mean(y)
                # добавляем новую центройду
                to.centroids1.append(centroid)
                if TEST_MODE:
                    if direction > 0:
                        print("вниз на 1 камере")
                    else:
                        print("вверх на 1 камере")
                    print(direction, centroid[1])
                # Смотрим вошел / вышел человек
                if not to.enter:
                    if TEST_MODE:
                        print("_________________")
                        if direction > 0:
                            print("dir TRUE")
                        if centroid[1] < self.door_points[1][1]:
                            print("2 TRUE")
                        if centroid[1] > self.door_points[0][1]:
                            print("3 TRUE")
                    if direction < 0 and centroid[1] < np.mean([self.line_1[1], self.line_1[3]]):
                        to.enter = True
                        # Удаляем информацию при выходе
                        # чтобы в след. раз это не влияло на вычисление среднего
                        if len(to.centroids2) > 0:
                            to.centroid2 = [to.centroids2[-1]]
                        if len(to.centroids2) > 0:
                            to.centroids1 = [to.centroids1[-1]]
                        self.totalPeopleCount += 1
                        self.db.set_is_in_building(str(to.id), str(1))
                        print("[1] Вошел:", to.id)
                elif to.enter:
                    if direction > 0 and centroid[1] > np.mean([self.line_1[1], self.line_1[3]]):
                        to.enter = False
                        # Удаляем информацию при выходе
                        # чтобы в след. раз это не влияло на вычисление среднего
                        if len(to.centroids2) > 0:
                            to.centroid2 = [to.centroids2[-1]]
                        if len(to.centroids2) > 0:
                            to.centroids1 = [to.centroids1[-1]]
                        self.totalPeopleCount -= 1
                        self.db.set_is_in_building(str(to.id), str(0))
                        print("[1] Вышел:", to.id)

            # сохраняем объект в словаре
            self.trackableObjects[objectID] = to
            # Рисукем на кадре ID объекта и центройду
            text = "ID: {}".format(self.db.get_name(str(objectID)))
            cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # перебираем людей на 2 кадре
        for (objectID, centroid) in objects2.items():
            # получаем объект по ID
            to = self.trackableObjects.get(objectID)

            # если этот объект не существует то создаем его
            if to is None:
                to = self.employee_dict[objectID]
                to.centroids2.append(centroid)
            # иначе определяем движется он вверх или вниз
            else:
                # Разница между текущей y координатой и средней центроиды
                # покажет направление (отрицательный - вверх; положительный - вниз)
                y = [c[1] for c in to.centroids2]
                direction = centroid[1] - np.mean(y)
                # добавляем новую центройду
                to.centroids1.append(centroid)
                if TEST_MODE:
                    if direction > 0:
                        print("вниз на 2 камере")
                    else:
                        print("вверх на 2 камере")
                    print(direction, centroid[1])
                # Смотрим вошел / вышел человек
                if not to.enter:
                    if TEST_MODE:
                        print("_________________")
                        if direction > 0:
                            print("dir TRUE")
                        if centroid[1] < self.door_points[1][1]:
                            print("2 TRUE")
                        if centroid[1] > self.door_points[0][1]:
                            print("3 TRUE")
                    if direction > 0 and centroid[1] > np.mean([self.line_2[1], self.line_2[3]]):
                        to.enter = True
                        # Удаляем информацию при выходе
                        # чтобы в след. раз это не влияло на вычисление среднего
                        if len(to.centroids2) > 0:
                            to.centroid2 = [to.centroids2[-1]]
                        if len(to.centroids2) > 0:
                            to.centroids1 = [to.centroids1[-1]]
                        self.totalPeopleCount += 1
                        self.db.set_is_in_building(str(to.id), str(1))
                        print("[2] Вошел:", to.id)
                elif to.enter:
                    if direction < 0 and centroid[1] < np.mean([self.line_2[1], self.line_2[3]]):
                        to.enter = False
                        # Удаляем информацию при выходе
                        # чтобы в след. раз это не влияло на вычисление среднего
                        if len(to.centroids2) > 0:
                            to.centroid2 = [to.centroids2[-1]]
                        if len(to.centroids2) > 0:
                            to.centroids1 = [to.centroids1[-1]]
                        self.totalPeopleCount -= 1
                        self.db.set_is_in_building(str(to.id), str(0))
                        print("[2] Вышел:", to.id)
            # сохраняем объект в словаре
            self.trackableObjects[objectID] = to
            # Рисукем на кадре ID объекта и центройду
            text = "ID: {}".format(self.db.get_name(str(objectID)))
            cv2.putText(frame2, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Проходимся по этому массиву и рисуем информацию на кадре
        text = "{}: {}".format("Inside", self.totalPeopleCount)
        cv2.putText(frame1, text, (10, self.H - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame2, text, (10, self.H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.totalFrames += 1

        return (frame1, frame2)