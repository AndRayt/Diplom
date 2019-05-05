from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import argparse
import pickle

# ТИП КЛАССИФИКАТОРА
_KNN = 0
_SVC = 1 # Отделение объектов разных классов гиперплоскостью

class FeatureVectorClassification:
    def __init__(self, classifier=_KNN):
        self.classifier = classifier

    def train_model(self,
                    output_file_path="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//dump_data//face_classifier.pickle",
                    output_labels_file_path="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//FaceNet//dump_data//labels.pickle",
                    employee_file_path="C://Users//Andrew//PycharmProjects//Diplom3_with_DB//recognition//employee_data//known_employees.pickle"):
        # загрузка векторов признаков лиц
        print("[INFO] загрузка признаков лиц...")
        employees = pickle.loads(open(employee_file_path, "rb").read())
        # кодирование меток
        print("[INFO] кодирование меток...")
        le = LabelEncoder()
        # записываем столько имен одинаковых подряд
        # сколько этому человеку соответсвует векторов
        names = [len(human.feature_vectors) * [human.id] for human in employees]
        new_names = []
        for el in names:
            for el_el in el:
                new_names.append(el_el)
        names = new_names
        labels = le.fit_transform(names)
        # тренировка модели (классификатор SVC) используя 128 векторы лиц
        print("[INFO] тренировка модели...")
        recognizer = None
        if self.classifier == _KNN:
            recognizer = KNeighborsClassifier(n_neighbors=5)
        elif self.classifier == _SVC:
            recognizer = SVC(C=1.0, kernel="linear", probability=True)
        else:
            raise Exception("Choose correct classifier! (_KNN / _SVC)")
        feautures = []
        for human in employees:
            for vector in human.feature_vectors:
                feautures.append(vector)
        recognizer.fit(feautures, labels)
        # записываем модель индефикации лица
        print("[INFO] сохранение обученной модели распознователя лиц FaceNet...")
        file = open(output_file_path, "wb")
        file.write(pickle.dumps(recognizer))
        file.close()
        # записываем метки на диск
        file = open(output_labels_file_path, "wb")
        file.write(pickle.dumps(le))
        file.close()

if __name__ == '__main__':
    classifier = FeatureVectorClassification()
    classifier.train_model()