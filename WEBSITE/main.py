from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("bank_churn_model.pkl")
le_card_type = joblib.load("le_card_type.pkl")
le_geography = joblib.load("le_geography.pkl")
le_gender = joblib.load("le_gender.pkl")

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict', response_class=HTMLResponse)
async def predict(
        request: Request,
        credit_score: int = Form(...),
        geography: str = Form(...),
        gender: str = Form(...),
        age: int = Form(...),
        tenure: int = Form(...),
        balance: float = Form(...),
        num_of_products: int = Form(...),
        has_cr_card: int = Form(0),
        is_active_member: int = Form(0),
        estimated_salary: float = Form(...),
        card_type: str = Form(...),
        point_earned: int = Form(...),
        satisfaction_score: int = Form(...)
):
    # Словарь
    data_dict = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Satisfaction Score": satisfaction_score,
        "Card Type": card_type,
        "Point Earned": point_earned
    }

    input_data = pd.DataFrame([data_dict])

    # Кодирование
    try:
        input_data['Geography'] = le_geography.transform(input_data['Geography'])
        input_data['Gender'] = le_gender.transform(input_data['Gender'])
        input_data['Card Type'] = le_card_type.transform(input_data['Card Type'])
    except Exception as e:
        return f"Ошибка кодирования: {e}"

    # Предсказание
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Уйдет" if prediction == 1 else "Останется"
    confidence = round(probability * 100, 2)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_text": f"Результат: {result} (Вероятность ухода: {confidence}%)"
    })