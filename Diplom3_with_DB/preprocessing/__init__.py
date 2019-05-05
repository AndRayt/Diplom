import imutils
import cv2

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
    CAMERA = 0
    vs = cv2.VideoCapture(CAMERA)
    ret_first, frame_first = vs.read()
    bs = BackgroundSubtraction(frame_first)
    while True:
        ret, frame = vs.read()
        result = bs.get_motion_box(frame)
        print(result)
