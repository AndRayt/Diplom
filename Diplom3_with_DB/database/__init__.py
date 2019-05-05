import sqlite3

class DataBase:
    def __init__(self):
        # Подключаемся (и создаем если еще не существует) к БД
        # Обязателен полный адресс к БД так как иначе скрипт будет работать только из данной директории
        self.connection = sqlite3.connect('C:\\Users\\Andrew\\PycharmProjects\\Diplom3\\database\\employee.db')
        # Объект курсора нужен для работы с БД
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute("""
                            CREATE TABLE employees (
                            id INTEGER primary key, 
                            name TEXT,
                            personal TEXT,
                            faces_images_dir TEXT,
                            enter_count INTEGER,
                            is_in_building BOOLEAN)
                            """)
        self.connection.commit()

    def del_table(self):
        self.cursor.execute("""
                            DROP TABLE employees
                            """)
        self.connection.commit()

    def add_employee(self, name, faces_images_dir, personal="NO DATA"):
        self.cursor.execute("INSERT INTO employees (name, personal, faces_images_dir, enter_count, is_in_building) VALUES (?, ?, ?, 0, 0)", (name, personal, faces_images_dir))
        self.connection.commit()

    # Возвращает данные в формате
    # (id, name, personal information, is in building?)
    def get_employee_lst(self):
        self.cursor.execute("""
                            SELECT * FROM employees
                            """)
        data = self.cursor.fetchall()
        return data

    # Возвращает имя человека по его id
    def get_name(self, id):
        self.cursor.execute("""
                                SELECT name FROM employees
                                WHERE id = ?;
                                """, (id))
        name = self.cursor.fetchall()
        return name[0][0]

    def set_is_in_building(self, id, is_in_building):
        self.cursor.execute("""
                                UPDATE employees
                                SET is_in_building = ?
                                WHERE id = ?;
                            """, (is_in_building, id))
        self.connection.commit()

    def delete_employee(self, id):
        self.cursor.execute("""
                                DELETE FROM employees
                                WHERE id = ?;
                            """, (id))
        self.connection.commit()

    def close_connection(self):
        self.cursor.close()
        self.connection.close()

if __name__ == '__main__':
    db = DataBase()
    db.del_table()
    db.create_table()
    db.close_connection()
