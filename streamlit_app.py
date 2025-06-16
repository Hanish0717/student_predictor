import streamlit as st
import joblib

model = joblib.load("pass_fail_model.pkl")

st.title("📚 Student Pass/Fail Predictor")

hours = st.number_input("Hours Studied", 0.0, 24.0, step=0.5)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, step=1.0)
assignments = st.selectbox("Assignments Completed?", ["Yes", "No"])
assignments = 1 if assignments == "Yes" else 0

if st.button("Predict"):
    prediction = model.predict([[hours, attendance, assignments]])
    result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"
    st.success(f"Prediction: {result}")
