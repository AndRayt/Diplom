from database import DataBase
from tkinter import *
from GUI.gui_add_person import GuiAddPerson
from GUI.gui_start import GUIStart
from GUI.gui_settings import GuiSettings
from main_2_cams_with_GUI import RECOGNITION_METHOD
from recognition.FaceNet.feature_extra import FeatureExtra
from recognition.FaceNet.train_model import FeatureVectorClassification
from recognition.EigenFaces import EigenFacesRecognition
from recognition.FisherFaces import FisherFacesRecognition
from recognition.LBPH import LBPHRecognition
from tracker.linking_with_2_face_recog import FACENET, EIGENFACES, FISHERFACES, LBPH
import pickle

LINE1_DATA = "C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\GUI\\data\\line1.pickle"
LINE2_DATA = "C:\\Users\\Andrew\\PycharmProjects\\Diplom3_with_DB\\GUI\\data\\line2.pickle"
# ПАРАМЕТРЫ ПО УМОЛЧАНИЮ
line_1 = (0, 0, 0, 0)
line_2 = (0, 0, 0, 0)

def set_employee_list(list_widget):
    list_widget.delete(0, END)
    for employee in db.get_employee_lst():
        in_building_str = "НЕИЗВЕСТНО"
        if employee[5] == 1:
            in_building_str = "ВНУТРИ ПРЕДПРИЯТИЯ"
        elif employee[5] == 0:
            in_building_str = "ВНЕ ПРЕДПРИЯТИЯ"
        format_str = "ID: {:3} |  ИМЯ: {:10} | {:18}\n".format(employee[0], employee[1], in_building_str)
        list_widget.insert(END, format_str)

db = DataBase()

root = Tk()
root.title("Система мониторинга сотрудников на предприятие | Райцын 15-АС")
root.geometry('685x420')
root.resizable(width=False, height=False)
frame_buttons = Frame(root)
frame_under_main = Frame(root)
frame_main = Frame(root)
frame_buttons.pack()
frame_under_main.pack()
frame_main.pack()

def button_add_listener():
    add_window = GuiAddPerson()
    root.withdraw()
    add_window.start(root, set_employee_list, listbox_employees)

# 47, 48
button_add = Button(frame_buttons, command = button_add_listener, text='ДОБАВИТЬ', bg='yellow', fg='green', width=31)

def button_settings_listener():
    settings_window1 = GuiSettings(1)
    settings_window1.start()

button_settings = Button(frame_buttons, command = button_settings_listener, text='1 CAM', width=15, bg='blue', fg='white')

def button_settings_listener_2():
    settings_window2 = GuiSettings(2)
    settings_window2.start()

button_settings_2 = Button(frame_buttons, command = button_settings_listener_2, text='2 CAM', width=15, bg='blue', fg='white')

def button_start_listener():
    global line_1, line_2
    try:
        if line_1 == (0, 0, 0, 0) and line_2 == (0, 0, 0, 0):
            print("[INFO] загрузка данных о координатах линий...")
            line_1 = pickle.loads(open(LINE1_DATA, "rb").read())
            line_2 = pickle.loads(open(LINE2_DATA, "rb").read())
    except Exception:
        print("[WARNING] неудалось загрузить данные о линиях")
        line_1 = (0, 0, 0, 0)
        line_2 = (0, 0, 0, 0)
    start_window = GUIStart(line_1, line_2)
    root.withdraw()
    start_window.start()

button_start = Button(frame_buttons, command=button_start_listener, text='НАЧАТЬ МОНИТОРИНГ', width=32, bg='green', fg='yellow')

label = Label(frame_under_main, text='СПИСОК СОТРУДНИКОВ', width=64, bg='pink', fg='red')

def button_delete_listener():
    slt = list(listbox_employees.curselection())
    for el in slt:
        db.delete_employee(str(el))
        listbox_employees.delete(el)
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

button_delete = Button(frame_under_main, command=button_delete_listener, text='УДАЛИТЬ', bg='red', fg='white', width=31)

scrollbar = Scrollbar(frame_main)
listbox_employees = Listbox(frame_main, width=480, height=150, yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)
text_employees = Text(frame_main, width=480, height=150, font='Helvetica22')
# Настройка данных в listbox
set_employee_list(listbox_employees)

button_add.grid(row=0, column=0)
button_settings.grid(row=0, column=1)
button_settings_2.grid(row=0, column=2)
button_start.grid(row=0, column=3)
label.grid(row=0, column=0)
button_delete.grid(row=0, column=1)
listbox_employees.pack()

root.mainloop()
