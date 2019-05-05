from database import DataBase

"""
    Класс содержащий информацию о работнике
"""
class Employee:
    def __init__(self, id, feature_vector=None):
        self.id = id
        self.feature_vectors = []
        # центройды 1 кадра
        self.centroids1 = []
        # центройды 2 кадра
        self.centroids2 = []
        # Вектор признаков из FaceNet
        self.feature_vectors.append(feature_vector)
        # Человек уже вошел в здание?
        self.enter = False

    def _create_note_in_db(self):
        db = DataBase()
        db.add_employee(id)
