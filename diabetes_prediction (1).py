import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Generate synthetic dataset resembling the Pima Indians Diabetes Dataset
# (Replace this section with `df = pd.read_csv('diabetes.csv')` if you have a real CSV file)
np.random.seed(42)
n_samples = 768

data = {
    'Pregnancies': np.random.randint(0, 15, n_samples),
    'Glucose': np.random.normal(120, 30, n_samples).clip(40, 200).astype(int),
    'BloodPressure': np.random.normal(70, 12, n_samples).clip(30, 122).astype(int),
    'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 99).astype(int),
    'Insulin': np.random.exponential(80, n_samples).clip(0, 846).astype(int),
    'BMI': np.round(np.random.normal(32, 7, n_samples).clip(15, 67), 1),
    'DiabetesPedigreeFunction': np.round(np.random.exponential(0.47, n_samples).clip(0.08, 2.42), 3),
    'Age': np.random.randint(21, 81, n_samples)
}

df = pd.DataFrame(data)

# Create a realistic target variable ('Outcome': 1 = Diabetic, 0 = Non-Diabetic)
risk_score = (
    (df['Glucose'] - 120) * 0.04 +
    (df['BMI'] - 30) * 0.08 +
    (df['Age'] - 35) * 0.03 +
    (df['DiabetesPedigreeFunction'] - 0.5) * 0.8 +
    np.random.normal(0, 1, n_samples)
)
df['Outcome'] = (risk_score > 0.5).astype(int)

# 2. Data Preprocessing
# Replace biologically impossible 0 values in certain columns with median values
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Predict on a New Patient Record
def predict_diabetes(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    patient_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                                columns=X.columns)
    patient_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    status = "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)"
    return status, probability

if __name__ == "__main__":
    sample_status, sample_prob = predict_diabetes(
        pregnancies=2, glucose=148, bp=72, skin_thickness=35,
        insulin=0, bmi=33.6, dpf=0.627, age=50
    )
    print(f"\nSample Patient Prediction: {sample_status} (Probability: {sample_prob:.2%})")
