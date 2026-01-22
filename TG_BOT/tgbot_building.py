
import asyncio
import joblib
import pandas as pd
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command


# Загружаем модель и энкодеры
model = joblib.load("bank_churn_model.pkl")
le_geo = joblib.load("le_geography.pkl")
le_gender = joblib.load("le_gender.pkl")
le_card = joblib.load("le_card_type.pkl")

API_TOKEN = '8529379362:AAGIrlN-M0oDVUtsNSpbSO8r6kghP1m1w34'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

class ChurnForm(StatesGroup):
    CreditScore = State()
    Geography = State()
    Gender = State()
    Age = State()
    Tenure = State()
    Balance = State()
    NumOfProducts = State()
    HasCrCard = State()
    IsActiveMember = State()
    EstimatedSalary = State()
    SatisfactionScore = State()
    CardType = State()
    PointEarned = State()

def make_kb(items):
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True, one_time_keyboard=True)


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await message.answer("Добро пожаловать! Предскажем отток клиента.\nВведите Credit Score (300-850):")
    await state.set_state(ChurnForm.CreditScore)

@router.message(ChurnForm.CreditScore)
async def process_score(message: Message, state: FSMContext):
    try:
        score = int(message.text)
        await state.update_data(CreditScore=score)
        await message.answer("Выберите страну:", reply_markup=make_kb(["France", "Germany", "Spain"]))
        await state.set_state(ChurnForm.Geography)
    except ValueError:
        await message.answer("Пожалуйста, введите число (например, 650).")

@router.message(ChurnForm.Geography)
async def process_geo(message: Message, state: FSMContext):
    try:
        # Кодируем страну через LabelEncoder
        geo_val = int(le_geo.transform([message.text])[0])
        await state.update_data(Geography=geo_val)
        await message.answer("Выберите пол:", reply_markup=make_kb(["Female", "Male"]))
        await state.set_state(ChurnForm.Gender)
    except Exception:
        await message.answer("Пожалуйста, выберите страну.")

@router.message(ChurnForm.Gender)
async def process_gender(message: Message, state: FSMContext):
    try:
        # Кодируем пол через LabelEncoder
        gender_val = int(le_gender.transform([message.text])[0])
        await state.update_data(Gender=gender_val)
        await message.answer("Введите возраст:", reply_markup=ReplyKeyboardRemove())
        await state.set_state(ChurnForm.Age)
    except Exception:
        await message.answer("Пожалуйста, выберите пол.")

@router.message(ChurnForm.Age)
async def process_age(message: Message, state: FSMContext):
    try:
        await state.update_data(Age=int(message.text))
        await message.answer("Сколько лет является клиентом?")
        await state.set_state(ChurnForm.Tenure)
    except ValueError:
        await message.answer("Введите возраст числом.")

@router.message(ChurnForm.Tenure)
async def process_tenure(message: Message, state: FSMContext):
    try:
        await state.update_data(Tenure=int(message.text))
        await message.answer("Баланс на счете (€):")
        await state.set_state(ChurnForm.Balance)
    except ValueError:
        await message.answer("Введите число лет.")

@router.message(ChurnForm.Balance)
async def process_balance(message: Message, state: FSMContext):
    try:
        await state.update_data(Balance=float(message.text))
        await message.answer("Количество используемых продуктов (1-4):")
        await state.set_state(ChurnForm.NumOfProducts)
    except ValueError:
        await message.answer("Введите сумму баланса числом.")

@router.message(ChurnForm.NumOfProducts)
async def process_products(message: Message, state: FSMContext):
    try:
        await state.update_data(NumOfProducts=int(message.text))
        await message.answer("Есть ли кредитная карта?", reply_markup=make_kb(["Да", "Нет"]))
        await state.set_state(ChurnForm.HasCrCard)
    except ValueError:
        await message.answer("Введите количество продуктов (число).")

@router.message(ChurnForm.HasCrCard)
async def process_card(message: Message, state: FSMContext):
    val = 1 if message.text == "Да" else 0
    await state.update_data(HasCrCard=val)
    await message.answer("Является ли активным участником?", reply_markup=make_kb(["Да", "Нет"]))
    await state.set_state(ChurnForm.IsActiveMember)

@router.message(ChurnForm.IsActiveMember)
async def process_active(message: Message, state: FSMContext):
    val = 1 if message.text == "Да" else 0
    await state.update_data(IsActiveMember=val)
    await message.answer("Приблизительная зарплата (€):", reply_markup=ReplyKeyboardRemove())
    await state.set_state(ChurnForm.EstimatedSalary)

@router.message(ChurnForm.EstimatedSalary)
async def process_salary(message: Message, state: FSMContext):
    try:
        await state.update_data(EstimatedSalary=float(message.text))
        await message.answer("Оценка удовлетворенности (1-5):")
        await state.set_state(ChurnForm.SatisfactionScore)
    except ValueError:
        await message.answer("Введите зарплату числом.")

@router.message(ChurnForm.SatisfactionScore)
async def process_satisfaction(message: Message, state: FSMContext):
    try:
        await state.update_data(SatisfactionScore=int(message.text))
        await message.answer("Тип карты:", reply_markup=make_kb(["DIAMOND", "GOLD", "PLATINUM", "SILVER"]))
        await state.set_state(ChurnForm.CardType)
    except ValueError:
        await message.answer("Введите число от 1 до 5.")

@router.message(ChurnForm.CardType)
async def process_card_type(message: Message, state: FSMContext):
    try:
        # Кодируем тип карты через LabelEncoder
        card_val = int(le_card.transform([message.text])[0])
        await state.update_data(CardType=card_val)
        await message.answer("Количество бонусных баллов (Points):", reply_markup=ReplyKeyboardRemove())
        await state.set_state(ChurnForm.PointEarned)
    except Exception:
        await message.answer("Пожалуйста, выберите тип карты кнопкой.")

@router.message(ChurnForm.PointEarned)
async def process_final(message: Message, state: FSMContext):
    try:
        await state.update_data(PointEarned=int(message.text))
        data = await state.get_data()
        df_input = pd.DataFrame([{
            'CreditScore': data['CreditScore'],
            'Geography': data['Geography'],
            'Gender': data['Gender'],
            'Age': data['Age'],
            'Tenure': data['Tenure'],
            'Balance': data['Balance'],
            'NumOfProducts': data['NumOfProducts'],
            'HasCrCard': data['HasCrCard'],
            'IsActiveMember': data['IsActiveMember'],
            'EstimatedSalary': data['EstimatedSalary'],
            'Satisfaction Score': data['SatisfactionScore'],
            'Card Type': data['CardType'],
            'Point Earned': data['PointEarned']
        }])

        # Предсказание
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1] * 100

        if prediction == 1:
            text = f"❌ Клиент склонен к уходу.\nВероятность оттока: {probability:.2f}%"
        else:
            text = f"✅ Клиент лоялен.\nВероятность ухода всего: {probability:.2f}%"

        await message.answer(text)
        await state.clear()
    except ValueError:
        await message.answer("Введите количество баллов числом.")

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
