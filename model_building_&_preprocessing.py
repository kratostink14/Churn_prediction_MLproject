# Дипломная работа

# Импортирую нужные библиотеки для работы
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Модель Случайного Леса для задач регрессии
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix  # Метрики для оценки качества модели
import kagglehub
import shap

pd.set_option('display.max_columns', None)

# 1. Загрузка датасета и обработка данных

# Данные скачиваю с kaggle, показываю путь(для себя)
path = kagglehub.dataset_download("radheshyamkollipara/bank-customer-churn")
print("Path to dataset files:", path)

# Загружаю датасет
data = pd.read_csv('Customer-Churn-Records.csv')

# Заранее удаляю колонки, которые не будут иметь никакого влияния на результат
df = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Кодирую столбцы с текстом в цифры
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_geography = LabelEncoder()
df['Geography'] = le_geography.fit_transform(df['Geography'])

le_card_type = LabelEncoder()
df['Card Type'] = le_card_type.fit_transform(df['Card Type'])

# Изучаю датасет
print(df.isnull().sum())  # Суммирование всех пустых значений
print(df.describe())  # Сводная статистика по числовым столбцам
# Вывод первых пяти строк
print(df.head())

# Делаю обратное преобразование, чтобы понять что за значения в текстовых столбцах
print("Кодировка Geography:", list(enumerate(le_geography.classes_)))
print("Кодировка Gender:", list(enumerate(le_gender.classes_)))
print("Кодировка Card Type:", list(enumerate(le_card_type.classes_)))

# Считаю сколько клиентов уже ушло и сколько осталось используя столбец Exited
print(df['Exited'].value_counts())
# Визуализирую
sns.countplot(x='Exited', data=df)
plt.title('Распределение оттока (0 = Остались, 1 = Ушли)')
plt.grid()
plt.savefig('Exited.png')
plt.show()

# Рисую тепловую карту корреляции между признаками, чтобы понять что от чего зависит
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True)
plt.title('Анализ корреляций между признаками')
plt.savefig('Correlation_heatmap.png')
plt.show()

# 2. Модель

# Разделяю данные на признаки и целевую переменную
X1 = df.drop(columns=['Exited'], axis=1)
y1 = df['Exited']

# Разделение данных на тренировочную и тестовую выборки
x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=25)

# Обучение модели Случайного леса
ch_modelv1 = RandomForestClassifier(n_estimators=25, random_state=42, max_depth=20, min_samples_split=8, min_samples_leaf=3)
ch_modelv1.fit(x_train1, y_train1)
# Предсказание на тестовой выборке
y_pred1 = ch_modelv1.predict(x_test1)

# Оценка модели случайного леса.
# Буду использовать метрики качества для модели классификации
accuracy1 = accuracy_score(y_test1, y_pred1)
print(f'Точность модели c колонкой Complain (Жалобы клиентов): {accuracy1*100:.2f}%')

roc_auc1 = roc_auc_score(y_test1, y_pred1)
print(f'ROC-AUC модели c колонкой Complain (Жалобы клиентов): {roc_auc1*100:.2f}%')

confusion_matrix1 = confusion_matrix(y_test1, y_pred1)
print('Матрица ошибок модели c колонкой Complain (Жалобы клиентов)')
print(confusion_matrix1)

# [[1604    6]
#  [   1  389]]

# По матрице ошибок видно, что модель правильно предсказала 1604 клиента останутся, а 389 уйдут
# Но модель так же допустила ошибку, что 6 уйдут, но они на самом деле остались и что 1 человек останется,
# но ушел

# Для эксперимента я попробую удалить колонку Complain - жалобы клиентов, так как модель думает,
# что, если есть жалоба клиент точно уйдет, это можно увидеть по анализу корреляции между признаками
# который я сделал раннее

# Разделяю данные на признаки и целевую переменную
X2 = df.drop(columns=['Exited', 'Complain'], axis=1)
y2 = df['Exited']

# Разделение данных на тренировочную и тестовую выборки
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=25)

# Обучение модели Случайного леса
ch_modelv2 = RandomForestClassifier(n_estimators=25, random_state=42, max_depth=20, min_samples_split=8, min_samples_leaf=3, class_weight='balanced')
ch_modelv2.fit(x_train2, y_train2)
# Предсказание на тестовой выборке
y_pred2 = ch_modelv2.predict(x_test2)
print(y_pred2)
# Оценка модели случайного леса.
# Буду использовать метрики качества для модели классификации
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f'Точность модели без колонки Complain (Жалобы клиентов): {accuracy2*100:.2f}%')

roc_auc2 = roc_auc_score(y_test2, y_pred2)
print(f'ROC-AUC модели без колонки Complain (Жалобы клиентов): {roc_auc2*100:.2f}%')

confusion_matrix2 = confusion_matrix(y_test2, y_pred2)
print('Матрица ошибок модели без колонки Complain (Жалобы клиентов)')
print(confusion_matrix2)

# Тут можно наблюдать меньше качества, но более реалистичные цифры,
# так как колонка Complain имевшая большее
# влияние в датасете удалена

# Сравнительная таблица метрик
metrics_df = pd.DataFrame({
    'Параметр': ['Accuracy (Точность)', 'ROC-AUC Score'],
    'Модель v1 (с Complain)': [f"{accuracy1*100:.2f}%", f"{roc_auc1*100:.2f}"],
    'Модель v2 (без Complain)': [f"{accuracy2*100:.2f}%", f"{roc_auc2*100:.2f}"]
})

# Вывод
print("Сравнительная таблица метрик качества моделей:")
print(metrics_df)

# 3. ТЕСТ

# Клиент, который останется
data_stay = {
    'CreditScore': 750,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 25,
    'Tenure': 10,
    'Balance': 0.0,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000.0,
    'Satisfaction Score': 5,
    'Card Type': 'DIAMOND',
    'Point Earned': 900
}

# Клиент, который уйдет
data_churn = {
    'CreditScore': 500,
    'Geography': 'Germany',
    'Gender': 'Female',
    'Age': 55,
    'Tenure': 1,
    'Balance': 140000.0,
    'NumOfProducts': 1,
    'HasCrCard': 0,
    'IsActiveMember': 0,
    'EstimatedSalary': 150000.0,
    'Satisfaction Score': 1,
    'Card Type': 'SILVER',
    'Point Earned': 200
}

# Выбор данных
current_test = data_stay

# Создаю DataFrame данных нового клиента
new_df = pd.DataFrame([current_test])

# Кодированные данные
new_df['Gender'] = le_gender.transform(new_df['Gender'])
new_df['Geography'] = le_geography.transform(new_df['Geography'])
new_df['Card Type'] = le_card_type.transform(new_df['Card Type'])

# Предсказание
pred = ch_modelv2.predict(new_df)
prob = ch_modelv2.predict_proba(new_df)[:, 1] # Вероятность ухода

# Результата
if pred[0] == 1:
    print(f"Результат: Клиент скорее всего УЙДЕТ (Вероятность: {prob[0]*100:.2f}%)")
else:
    print(f"Результат: Клиент ОСТАНЕТСЯ (Вероятность ухода всего: {prob[0]*100:.2f}%)")

# Сохранение модели и кодированных данных
import joblib

joblib.dump(ch_modelv2, "bank_churn_model.pkl")
print("Модель сохранена в файл bank_churn_model.pkl")

joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_geography, 'le_geography.pkl')
joblib.dump(le_card_type, 'le_card_type.pkl')
print('Кодированные данные сохранены')





