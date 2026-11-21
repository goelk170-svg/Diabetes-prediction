import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered"
)


@st.cache_resource
def load_and_train_model():
    np.random.seed(42)
    n_samples = 768

    df = pd.DataFrame({
        "Pregnancies": np.random.randint(0, 15, n_samples),
        "Glucose": np.random.normal(120, 30, n_samples).clip(40, 200).astype(int),
        "BloodPressure": (
            np.random.normal(70, 12, n_samples).clip(30, 122).astype(int)
        ),
        "SkinThickness": (
            np.random.normal(20, 15, n_samples).clip(0, 99).astype(int)
        ),
        "Insulin": np.random.exponential(80, n_samples).clip(0, 846).astype(int),
        "BMI": np.round(np.random.normal(32, 7, n_samples).clip(15, 67), 1),
        "DiabetesPedigreeFunction": np.round(
            np.random.exponential(0.47, n_samples).clip(0.08, 2.42), 3
        ),
        "Age": np.random.randint(21, 81, n_samples),
    })

    risk_score = (
        (df["Glucose"] - 120) * 0.04
        + (df["BMI"] - 30) * 0.08
        + (df["Age"] - 35) * 0.03
        + (df["DiabetesPedigreeFunction"] - 0.5) * 0.8
        + np.random.normal(0, 1, n_samples)
    )
    df["Outcome"] = (risk_score > 0.5).astype(int)

    cols_with_zeros = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, list(X.columns)


model, scaler, feature_names = load_and_train_model()

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "Enter the patient's diagnostic measurements below to estimate diabetes"
    " risk."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=2, step=1
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=200,
            value=70,
            step=1,
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=0.0,
            max_value=70.0,
            value=28.5,
            step=0.1,
            format="%.1f",
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.47,
            step=0.01,
            format="%.3f",
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=35, step=1
        )

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
        ]],
        columns=feature_names,
    )
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(
            "**High Risk (Positive)** — Estimated Probability:"
            f" **{probability:.1%}**"
        )
    else:
        st.success(
            "**Low Risk (Negative)** — Estimated Probability:"
            f" **{probability:.1%}**"
        )

    st.caption(
        "Disclaimer: This tool is for educational and informational purposes"
        " only and is not a medical diagnosis."
    )
