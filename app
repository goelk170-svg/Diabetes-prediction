import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction")
st.write("Enter the patient details below to get a prediction from the machine learning model.")

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "diabetes_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please place diabetes_model.pkl in the same folder as app.py.")
    st.stop()

st.subheader("Patient Information")

age = st.number_input("Age", min_value=18, max_value=80, value=45)
gender_label = st.selectbox("Gender", ["Female", "Male", "Unknown"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5)
glucose = st.number_input("Glucose", min_value=40.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=125)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=500, value=210)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=110)
hba1c = st.number_input("HbA1c", min_value=3.0, max_value=20.0, value=6.0)
smoking_label = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
activity_label = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
family_label = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# Convert categorical values into numbers
gender_map = {"Female": 0, "Male": 1, "Unknown": 2}
smoking_map = {"Never": 0, "Former": 1, "Current": 2}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}
family_map = {"No": 0, "Yes": 1}

# Prepare patient data in the same feature order used during training
patient = pd.DataFrame([{
    "Age": age,
    "Gender": gender_map[gender_label],
    "BMI": bmi,
    "Glucose": glucose,
    "Blood_Pressure": blood_pressure,
    "Cholesterol": cholesterol,
    "Insulin": insulin,
    "HbA1c": hba1c,
    "Smoking_Status": smoking_map[smoking_label],
    "Physical_Activity": activity_map[activity_label],
    "Family_History": family_map[family_label]
}])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    prediction = int(model.predict(patient)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Model Prediction: Diabetes Yes")
    else:
        st.success("Model Prediction: Diabetes No")

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probability = model.predict_proba(patient)[0][classes.index(1)]
        st.metric("Estimated Probability of Diabetes", f"{probability * 100:.2f}%")

st.divider()
st.caption("Educational project only. This model is not a medical diagnostic tool and should not be used for medical decisions.")
