import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient health metrics below to predict diabetes risk.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=29.4, step=0.1)
    glucose = st.number_input("Glucose Level", min_value=50.0, max_value=300.0, value=140.0, step=1.0)
    bp = st.number_input("Blood Pressure", min_value=60, max_value=220, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

with col2:
    insulin = st.number_input("Insulin", min_value=10, max_value=400, value=150)
    hba1c = st.number_input("HbA1c", min_value=4.0, max_value=15.0, value=7.5, step=0.1)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# Encode inputs identically to your notebook preprocessing
gender_map = {"Female": 0, "Male": 1, "Unknown": 2}
smoking_map = {"Never": 0, "Former": 1, "Current": 2}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}
history_map = {"No": 0, "Yes": 1}

if st.button("Predict Diabetes Risk", type="primary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_map[gender],
        "BMI": bmi,
        "Glucose": glucose,
        "Blood_Pressure": bp,
        "Cholesterol": cholesterol,
        "Insulin": insulin,
        "HbA1c": hba1c,
        "Smoking_Status": smoking_map[smoking],
        "Physical_Activity": activity_map[activity],
        "Family_History": history_map[family_history]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"**High Risk of Diabetes** (Probability: {probability:.1%})")
    else:
        st.success(f"**Low Risk of Diabetes** (Probability: {probability:.1%})")
