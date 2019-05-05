from detector.haar_cascades_detection import *
from entity.employee import Employee
import os
import pickle
from imutils import paths
from database import DataBase
from preprocessing import frame_preprocessing
"""
    Класс извлекает вектор признаков для каждого известного лица
    используя нейронную сеть FaceNet
"""

# ДЕТЕКТОРЫ
HAAR_CASCADES = 0
FACE_DETECTION = 1
NN_DETECTION = 2

class FeatureExtra:
    def __init__(self, detector_type=HAAR_CASCADES,
                 recognition_network="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//openface_nn4.small2.v1.t7"):
        # загрузка детектора лиц
        print("[INFO] загрузка дектора лиц...")
        self.detector_type = detector_type
        self.detector = None
        if detector_type == HAAR_CASCADES:
            self.detector = HaarCascadesDetection(mode=HaarCascadesDetection.FACE)
        elif detector_type == FACE_DETECTION:
            pass
        elif detector_type == NN_DETECTION:
            pass
        else:
            raise Exception("Error! Detector type dont support!")
        # загрузка FaceNet
        print("[INFO] загрузчик определителя лиц...")
        self.embedder = cv2.dnn.readNetFromTorch(recognition_network)

    def extract(self,
                file_path="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//employee_data//known_employees.pickle",
                confidence=0.5):
        print("[INFO] начинается процесс извлечения признаков лиц...")
        # загрузка обучающего набора
        print("[INFO] количественное определение лиц...")
        # инициализация массива распознаных работников
        employees = dict() # словарь уже известных сотрудников
        db = DataBase()
        employees_lst = db.get_employee_lst()
        id_faces_lst = [] # лист с кортежами (id, faces_dir)
        for employee in employees_lst:
            id_faces_lst.append((employee[0], employee[3]))
        for employee in id_faces_lst:
            id = employee[0]
            for imagePath in list(paths.list_images(employee[1])):
                image = cv2.imread(imagePath)
                image = frame_preprocessing(image, gray=False)
                (h, w) = image.shape[:2]
                # обнаруживаем и локализуем лица
                # создаем блоб
                image = cv2.resize(image, (320, 480))
                # пропускаем блобы через дектор и получаем список обнаружения
                detections = []
                if self.detector_type == HAAR_CASCADES:
                    detections = self.detector.human_detection(image)
                elif self.detector_type == FACE_DETECTION:
                    detections = self.detector.human_detection(image, confidence)
                elif self.detector_type == NN_DETECTION:
                    detections = self.detector.human_detection(image, w, h, confidence)
                # считаем что в обучающей выборке может быть лишь одно лицо
                # поэтому берем первый задетектированный бокс
                if len(detections) > 0:
                    startX, startY, endX, endY = detections[0]
                    # извлекаем ROI лица
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    # проверяем что размеры рамки достаточно велики
                    if fW < 20 or fH < 20:
                        continue
                    # построим блоб из области лица
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    # посылаем его в сверточную сеть и получаем 128 вектор
                    # который описывает лицо
                    self.embedder.setInput(faceBlob)
                    feature_vector = self.embedder.forward()
                    # создаем объект нового работника
                    # для корректной работы классификатора необходимо чтобы вектор признаков был одномерный
                    # поэтому используется numpy flatten()
                    if id in employees.keys():
                        employees[id].feature_vectors.append(feature_vector.flatten())
                    else:
                        human = Employee(id, feature_vector.flatten())
                        employees[id] = human

        # Сохраняем словарь работников на диск
        print("[INFO] сериализация данных...")
        file = open(file_path, "wb")
        file.write(pickle.dumps(list(employees.values())))
        file.close()

if __name__ == '__main__':
    feObject = FeatureExtra()
    feObject.extract()









