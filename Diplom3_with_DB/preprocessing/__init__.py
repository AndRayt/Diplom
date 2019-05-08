import imutils
import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy import ndimage

from detector.haar_cascades_detection import HaarCascadesDetection

TEST_ON = False

"""
    Метод для предобработки кадров
"""
def frame_preprocessing(frame, resize_x=400, resize_y=400, gray=True):
    frame = cv2.resize(frame, (resize_x, resize_y))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Histogram Equalize
    frame = cv2.equalizeHist(frame)
    if not gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Фильтр гаусса для удаления шумов
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

"""
    Метод выравнивает лицо по линии глаз
"""
def alignment_face(img_face):
    def alignment(img, face_box):
        predictor_path = "C:\\Users\\Andrew\\PycharmProjects\\Diplom2\\shape_predictor_5_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        points = predictor(img, face_box)
        points = face_utils.shape_to_np(points)
        return points

    rectangle = dlib.rectangle(0, 0, img_face.shape[1], img_face.shape[0])
    points = alignment(img_face, rectangle)
    # Выравнивание фотографии по линии глаз
    right_eye = [(points[0][0] + points[1][0]) / 2,
                 (points[0][1] + points[1][1]) / 2]  # крайняя правая точка правого глаза плюс крайняя левая по полам
    left_eye = [(points[2][0] + points[3][0]) / 2,
                (points[2][1] + points[3][1]) / 2]  # крайняя левая точка левого глаза плюс крайняя права по полам
    # Угол между глаз - тангес отношение Y / X
    y = right_eye[1] - left_eye[1]
    x = right_eye[0] - left_eye[0]
    ang = np.degrees(np.arctan2(y, x))
    # Поворачиваем изображение
    rotated_image = ndimage.rotate(img_face, ang)
    #cv2.imshow("face", rotated_image)
    #cv2.waitKey()
    return rotated_image

"""
    Вычитание фона
    Пока не ипользуется
"""
class BackgroundSubtraction:
    def __init__(self, first_frame, min_area_size=500):
        self.first_frame = frame_preprocessing(first_frame)
        if TEST_ON: cv2.imshow("[TEST] Background", self.first_frame)
        self.min_area_size = min_area_size # минимальный размер бокса

    """
        Метод реализующий вычитание фона
        возвращает координаты бокса где происходит движение 
        confidence - игнорируемая разность интенсивности
    """
    def get_motion_box(self, frame, confidence=1):
        # Результат - массив боксов с движущимися людьми
        result = []
        # Предобработка frame
        frame = frame_preprocessing(frame)
        # Разница между фоном и текущем кадром
        # (Вычитание интенсивность по каждому пикселю)
        frameDelta = cv2.absdiff(self.first_frame, frame)
        #if TEST_ON: cv2.imshow("[TEST] Delta", frameDelta)
        # Убираем шумы
        # (Бинаризуем результат с указанным порогом точности)
        # if confidence > 255: confidence = 255
        # elif confidence < 0: confidence = 0
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # Расширяем изображение чтобы заполнить дыры
        thresh = cv2.dilate(thresh, None, iterations=10)
        #if TEST_ON:
        #    cv2.imshow("[TEST] Thresh", thresh)
        #    cv2.imshow("[TEST] Frame", frame)
        # Находим контуры движущегося объекта
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # Получаем массив контуров
        cnts = imutils.grab_contours(cnts)
        # Перебираем все контуры
        for c in cnts:
            # Если площадь фигуры слишком мала то игнорируем
            if cv2.contourArea(c) < self.min_area_size:
                continue
            # Получаем координаты бокса объекта
            (x, y, w, h) = cv2.boundingRect(c)
            # Преобразуем результат в формат (startX, startY, endX, endY)
            startX = x
            startY = y
            endX = x + w
            endY = y + h
            result.append((startX, startY, endX, endY))
        return result

# TEST_MODULE
if __name__ == '__main__':
    #CAMERA = 0
    #vs = cv2.VideoCapture(CAMERA)
    #ret_first, frame_first = vs.read()
    #bs = BackgroundSubtraction(frame_first)
    #while True:
    #    ret, frame = vs.read()
    #    result = bs.get_motion_box(frame)
    #    print(result)
    image_path = "C:\\Users\\Andrew\\PycharmProjects\\Diplom2\\jenna_al.jpg"
    detector = HaarCascadesDetection()
    img = cv2.imread(image_path)
    boxes = detector.human_detection(img)
    box = boxes[0]
    rectangle = dlib.rectangle(box[0], box[1], box[2], box[3])
    img = img[box[1]:box[3], box[0]:box[2]]
    al_face = alignment_face(img)
    cv2.imshow("sd", al_face)
    cv2.waitKey()