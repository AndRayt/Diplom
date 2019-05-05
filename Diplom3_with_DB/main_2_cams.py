from preprocessing import frame_preprocessing
from tracker.linking_with_2_face_recog import *
from entity.employee import Employee
from detector.haar_cascades_detection import HaarCascadesDetection

from collections import OrderedDict
from imutils.video import FPS
import numpy as np
import time
import dlib
import cv2
import pickle

TEST_MODE = False # Вывод тестовой информации
CAMERA1 = 0 # Номер камеры 1
CAMERA2 = 1 # Номер камеры 2
SKIP_FRAMES = 30 # Количество пропускаемых между детектированиями кадров

print("[INFO] начинаем видеопоток...")
vs1 = cv2.VideoCapture(CAMERA1)
time.sleep(2.0)
vs2 = cv2.VideoCapture(CAMERA2)
time.sleep(2.0)

# переменные для высоты и ширины кадра
W = None
H = None

# инициализируем трекер центроид
linking_tracker = SimpleLinkingWithRecog(recog_method=FISHERFACES)
# Инициализируем детектор
detector = HaarCascadesDetection(mode=HaarCascadesDetection.FACE)
trackers1 = []
trackers2 = []
# Словарь вида objectID : TrackableOjbect
trackableObjects = {}

# Количество обработанных кадров
totalFrames = 0
# Подсчет людей внутри предприятия
totalPeopleCount = 0
# Счетчик фпс
fps = FPS().start()
# Параметры бокса двери
door_points = ((260, 10), (380, 170))

# Получаем список известных работников
print("[INFO] загрузка зарегестрирвоанных работников в системе...")
employees_list = pickle.loads(open("C://Users//Andrew//PycharmProjects//Diplom3//recognition//FaceNet//dump_data//known_employees.pickle", "rb").read())
# Переводим лист в словарь по ID
employee_dict = OrderedDict()
for employee in employees_list:
	employee_dict[employee.id] = employee

# Обрабатываем кадры
while True:
	# Считываем кадр с 1 камеры
	ret1, frame1 = vs1.read()
	ret2, frame2 = vs2.read()

	# Предобработка кадра для dlib
	gray_frame1 = frame_preprocessing(frame1)
	gray_frame2 = frame_preprocessing(frame2)

	# Устанавливаем H W
	if W is None or H is None:
		(H, W) = frame1.shape[:2]

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
	if totalFrames % SKIP_FRAMES == 20:
		trackers1, trackers2 = [], []
		# Начинаем обнаружение
		status = "Detecting"
		faceTrackers = []
		# Получаем массив боксов людей в кадре 1
		list_human_box1 = detector.human_detection(frame1)
		# Получаем массив боксов людей в кадре 2
		list_human_box2 = detector.human_detection(frame2)
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
	else:
		# проходим через все трекеры
		for tracker in trackers1:
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

		for tracker in trackers2:
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

	# Линия относительно которой считается что люди идут вверх или вниз
	#cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
	cv2.rectangle(frame1, door_points[0], door_points[1], (0, 255, 255))

	# связываем старые центроиды с новыми вычисленными
	objects1, objects2 = linking_tracker.update(rects1, frame1, rects2, frame2)
	if TEST_MODE: print("obj1:", objects1, "obj2", objects2)

	# определяем человек пошел вверх или вниз
	# перебираем людей на 1 кадре
	for (objectID, centroid) in objects1.items():
		# получаем объект по ID
		to = trackableObjects.get(objectID)

		# если этот объект не существует то создаем его
		if to is None:
			to = employee_dict[objectID]
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
					print("вниз")
				else:
					print("вверх")
				print(direction, centroid[1])
			# Смотрим вошел / вышел человек
			if not to.enter:
				if TEST_MODE:
					print("_________________")
					if direction > 0:
						print("dir TRUE")
					if centroid[1] < door_points[1][1]:
						print("2 TRUE")
					if centroid[1] > door_points[0][1]:
						print("3 TRUE")
				if direction > 0 and centroid[1] < door_points[1][1] and centroid[1] > door_points[0][1]:
					to.enter = True
					to.centroids1 = [to.centroids1[-1]]
					totalPeopleCount += 1
					print("Вошел:", to.id)
			elif to.enter:
				if direction < 0 and centroid[1] < door_points[1][1] and centroid[1] > door_points[0][1]:
					to.enter = False
					to.centroids1 = [to.centroids1[-1]]
					totalPeopleCount -= 1
					print("Вышел:", to.id)

		# сохраняем объект в словаре
		trackableObjects[objectID] = to
		#print(objectID)
		# Рисукем на кадре ID объекта и центройду
		text = "ID {}".format(objectID)
		cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# перебираем людей на 2 кадре
	for (objectID, centroid) in objects2.items():
		# получаем объект по ID
		to = trackableObjects.get(objectID)

		# если этот объект не существует то создаем его
		if to is None:
			to = employee_dict[objectID]
			to.centroids2.append(centroid)

		# сохраняем объект в словаре
		trackableObjects[objectID] = to
		# Рисукем на кадре ID объекта и центройду
		text = "ID {}".format(objectID)
		cv2.putText(frame2, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame2, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# Проходимся по этому массиву и рисуем информацию на кадре
	text = "{}: {}".format("Inside", totalPeopleCount)
	cv2.putText(frame1, text, (10, H - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# показываем кадр на экране
	cv2.imshow("Frame1", frame1)
	cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF

	# q - закрыть программу
	if key == ord("q"):
		break

	# обновляем счетчик кадров
	totalFrames += 1
	fps.update()

# останавливаем счетчик кадров и показываем информацию
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# закрываем Окно
vs1.release()
vs2.release()
cv2.destroyAllWindows()