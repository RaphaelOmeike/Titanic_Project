import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = 'model/titanic_survival_model.joblib'
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found! Please make sure 'model/titanic_survival_model.joblib' exists.")
        return None

model = load_model()

# Title and Description
st.title("🚢 Titanic Survival Prediction System")
st.markdown("""
    Welcome to the Titanic Survival Prediction System. 
    Enter the passenger details below to predict their survival probability.
""")

# Input Form
st.sidebar.header("Passenger Details")

def user_input_features():
    pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
    sex = st.sidebar.radio("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 100, 30)
    fare = st.sidebar.number_input("Ticket Fare", min_value=0.0, value=30.0, step=0.1)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: "Southampton" if x=="S" else ("Cherbourg" if x=="C" else "Queenstown"))

    data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'Fare': fare,
        'Embarked': embarked
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Panel Display
st.subheader("Passenger Information")
st.write(input_df)

if st.button("Predict Survival"):
    if model:
        # Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        survived = prediction[0] == 1
        probability = prediction_proba[0][1] if survived else prediction_proba[0][0]

        st.subheader("Prediction Result")
        if survived:
            st.success(f"🎉 **Survived** (Confidence: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"💀 **Did Not Survive** (Confidence: {probability:.2%})")
    else:
        st.error("Model is not loaded. Cannot predict.")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name] | Matric No: [Your Matric No]")
