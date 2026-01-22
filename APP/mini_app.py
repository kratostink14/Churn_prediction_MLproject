import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# 1. Загрузка модели и энкодеров
def resource_path(relative_path):
    """ Получает путь к файлу, работает и в Python, и в .exe """
    try:
        # PyInstaller создает временную папку _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Загрузка данных
try:
    model = joblib.load(resource_path("bank_churn_model.pkl"))
    le_gender = joblib.load(resource_path("le_gender.pkl"))
    le_geography = joblib.load(resource_path("le_geography.pkl"))
    le_card_type = joblib.load(resource_path("le_card_type.pkl"))
except Exception as e:
    # Создаем временное окно для вывода ошибки, так как основное еще не создано
    temp_root = tk.Tk()
    temp_root.withdraw()
    messagebox.showerror("Критическая ошибка", f"Не удалось загрузить файлы моделей!\n{e}")
    sys.exit(1) # Жестко прекращаем работу, если данные не загружены


class ChurnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Customer Churn Predictor")
        self.root.geometry("500x750")
        self.root.configure(padx=20, pady=20)

        # Заголовок
        ttk.Label(root, text="Прогноз оттока клиента", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                        pady=10)

        # ЛИЧНЫЕ ДАННЫЕ
        self.create_label_section("Личные данные", 1)

        self.age = self.create_input("Возраст:", 2, "42")
        self.gender = self.create_combo("Пол:", 3, ["Male", "Female"])
        self.geo = self.create_combo("Страна:", 4, ["France", "Germany", "Spain"])
        self.tenure = self.create_input("Лет в банке (Tenure):", 5, "5")

        # create_input - позволяет пользователю ввести параметр самому
        # create_combo - позволяет пользователю выбрать между вариантами, а не вводить параметры самому если параметры ограничиваются двумя или более

        # --- СЕКЦИЯ: ФИНАНСЫ ---
        self.create_label_section("Финансовые показатели", 6)

        self.credit = self.create_input("Кредитный рейтинг:", 7, "650")
        self.balance = self.create_input("Баланс счета:", 8, "125000")
        self.salary = self.create_input("Зарплата:", 9, "50000")
        self.card_type = self.create_combo("Тип карты:", 10, ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
        self.points = self.create_input("Бонусные баллы:", 11, "450")

        # --- СЕКЦИЯ: АКТИВНОСТЬ ---
        self.create_label_section("Активность", 12)

        self.products = self.create_input("Кол-во продуктов:", 13, "2")
        self.satisfaction = self.create_input("Индекс довольства (1-5):", 14, "3")

        self.has_card = tk.IntVar(value=1)
        ttk.Checkbutton(root, text="Есть кредитная карта", variable=self.has_card).grid(row=15, column=0, columnspan=2,
                                                                                        sticky="w")

        self.is_active = tk.IntVar(value=0)
        ttk.Checkbutton(root, text="Активный участник", variable=self.is_active).grid(row=16, column=0, columnspan=2,
                                                                                      sticky="w")

        # Кнопка расчета
        self.btn_predict = ttk.Button(root, text="РАССЧИТАТЬ ПРОГНОЗ", command=self.predict)
        self.btn_predict.grid(row=17, column=0, columnspan=2, pady=30)

    def create_label_section(self, text, row):
        label = ttk.Label(self.root, text=text, font=("Arial", 10, "italic", "underline"), foreground="gray")
        label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(15, 5))

    def create_input(self, text, row, default):
        ttk.Label(self.root, text=text).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(self.root)
        entry.insert(0, default)
        entry.grid(row=row, column=1, pady=5, sticky="e")
        return entry

    def create_combo(self, text, row, values):
        ttk.Label(self.root, text=text).grid(row=row, column=0, sticky="w")
        combo = ttk.Combobox(self.root, values=values, state="readonly")
        combo.set(values[0])
        combo.grid(row=row, column=1, pady=5, sticky="e")
        return combo

    def predict(self):
        try:
            # 1. Собираем данные из интерфейса
            input_data = {
                'CreditScore': int(self.credit.get()),
                'Geography': self.geo.get(),
                'Gender': self.gender.get(),
                'Age': int(self.age.get()),
                'Tenure': int(self.tenure.get()),
                'Balance': float(self.balance.get()),
                'NumOfProducts': int(self.products.get()),
                'HasCrCard': int(self.has_card.get()),
                'IsActiveMember': int(self.is_active.get()),
                'EstimatedSalary': float(self.salary.get()),
                'Satisfaction Score': int(self.satisfaction.get()),
                'Card Type': self.card_type.get(),
                'Point Earned': int(self.points.get())
            }

            # 2. Превращаем в DataFrame
            df = pd.DataFrame([input_data])

            # 3. Кодируем признаки
            df['Gender'] = le_gender.transform(df['Gender'])
            df['Geography'] = le_geography.transform(df['Geography'])
            df['Card Type'] = le_card_type.transform(df['Card Type'])

            # 4. Прогноз
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]

            # 5. Вывод результата
            if prediction == 1:
                result_text = f"КЛИЕНТ УЙДЕТ\n(Вероятность: {probability:.2%})"
                color = "red"
            else:
                result_text = f"КЛИЕНТ ОСТАНЕТСЯ\n(Вероятность ухода: {probability:.2%})"
                color = "green"

            messagebox.showinfo("Результат анализа", result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Проверьте правильность введенных данных!\nДетали: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnApp(root)
    root.mainloop()