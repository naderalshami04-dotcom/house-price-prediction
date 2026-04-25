import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 🔹 Load Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# 🔹 Title
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("أدخل البيانات الأساسية وسيتم التنبؤ بالسعر")

# -----------------------------
# 🔹 INPUTS (User Friendly)
# -----------------------------

area = st.number_input("Area (m²) - مساحة البيت", min_value=0)

total_rooms = st.number_input("Total Rooms - عدد الغرف", min_value=1)

furnishing = st.selectbox(
    "Furnishing Status - حالة الفرش",
    ["غير مفروش", "نصف مفروش", "مفروش بالكامل"]
)

furnishing_map = {
    "غير مفروش": 0,
    "نصف مفروش": 1,
    "مفروش بالكامل": 2
}

furnishing_value = furnishing_map[furnishing]

# -----------------------------
# 🔹 PREDICTION
# -----------------------------
if st.button("Predict"):

    # كل features اللي اتدرب عليها الموديل
    features = list(model.feature_names_in_)

    # إنشاء DataFrame بنفس شكل التدريب
    input_data = pd.DataFrame(0, index=[0], columns=features)

    # تعبئة القيم الأساسية
    if "area" in input_data.columns:
        input_data["area"] = area

    if "total_rooms" in input_data.columns:
        input_data["total_rooms"] = total_rooms

    if "furnishingstatus" in input_data.columns:
        input_data["furnishingstatus"] = furnishing_value

    # التنبؤ (log price)
    prediction = model.predict(input_data)

    # تحويل من log إلى سعر حقيقي
    real_price = np.exp(prediction[0])

    # عرض النتيجة
    st.success(f"🏠 Predicted Price = {real_price:,.0f}")