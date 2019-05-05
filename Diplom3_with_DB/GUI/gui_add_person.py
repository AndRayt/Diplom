from database import DataBase
from tkinter import *
from recognition.FaceNet.feature_extra import FeatureExtra
from recognition.FaceNet.train_model import FeatureVectorClassification
from recognition.EigenFaces import EigenFacesRecognition
from recognition.FisherFaces import FisherFacesRecognition
from recognition.LBPH import LBPHRecognition
from tracker.linking_with_2_face_recog import FACENET, EIGENFACES, FISHERFACES, LBPH
from main_2_cams_with_GUI import RECOGNITION_METHOD

class GuiAddPerson:
    def start(self, old_root, function, widget):
        root = Tk()
        frame1 = Frame(root)
        frame1.pack()
        root.geometry('400x100')
        root.title("Добавить пользователя в систему")

        label_name = Label(frame1, text="Имя сотрудника:")
        entry_name = Entry(frame1, width=35)
        label_name.grid(row=0, column=0)
        entry_name.grid(row=0, column=1)
        label_info = Label(frame1, text="Дополнительная информация:")
        entry_info = Entry(frame1, width=35)
        label_info.grid(row=1, column=0)
        entry_info.grid(row=1, column=1)
        label_photo = Label(frame1, text="Адрес папки с фотографиями:")
        entry_photo = Entry(frame1, width=35)
        label_photo.grid(row=2, column=0)
        entry_photo.grid(row=2, column=1)

        def button_listener():
            db = DataBase()
            db.add_employee(str(entry_name.get()), str(entry_photo.get()), str(entry_info.get()))
            db.close_connection()
            function(widget)
            # запускаем обучение модели распознования
            if RECOGNITION_METHOD == FACENET:
                feObject = FeatureExtra()
                feObject.extract()
                classifier = FeatureVectorClassification()
                classifier.train_model()
            elif RECOGNITION_METHOD == EIGENFACES:
                rec = EigenFacesRecognition()
                rec.traning()
            elif RECOGNITION_METHOD == FISHERFACES:
                rec = FisherFacesRecognition()
                rec.traning()
            elif RECOGNITION_METHOD == LBPH:
                rec = LBPHRecognition()
                rec.traning()
            else:
                raise Exception("Select correct recognition method")
            old_root.deiconify()
            root.destroy()

        button = Button(root, command=button_listener, text="ДОБАВИТЬ В СИСТЕМУ", bg="green", fg="white", width=55)
        button.pack()

        root.mainloop()