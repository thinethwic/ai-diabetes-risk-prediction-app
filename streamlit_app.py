import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Model Loading ---
# The @st.cache_resource decorator ensures the model is loaded only once,
# which is crucial for performance in a Streamlit app.
MODEL_FILE = 'diabetes.joblib'

@st.cache_resource
def load_model():
    """Load the pre-trained scikit-learn pipeline."""
    # Check if the model file exists in the current directory (where Streamlit is run)
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' not found.")
        st.stop()
    try:
        # Load the saved model pipeline
        pipeline = joblib.load(MODEL_FILE)
        return pipeline
    except Exception as e:
        st.error(f"Error loading the model pipeline: {e}")
        st.stop()

# Load the model globally
model_pipeline = load_model()

# --- Application UI Setup ---
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("Use the controls below to input a patient's characteristics and predict their risk of being **Diabetic (Outcome = 1)** or **Non-Diabetic (Outcome = 0)**.")
st.divider()

# --- Sidebar for Inputs ---
st.sidebar.header("Patient Input Features")

# Helper function to get inputs with sensible defaults and ranges from the notebook data
def user_input_features():
    """Creates input widgets for all required features."""
    
    # Features are based on the model in Diabetes.ipynb: Glucose, BloodPressure, Insulin, BMI, DPF, Age.
    
    glucose = st.sidebar.number_input(
        'Glucose (mg/dL)', 
        min_value=0.0, max_value=200.0, 
        value=120.0, step=1.0, 
        format="%.1f",
        help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test."
    )
    
    blood_pressure = st.sidebar.number_input(
        'Blood Pressure (mmHg)', 
        min_value=0.0, max_value=130.0, 
        value=70.0, step=1.0, 
        format="%.1f",
        help="Diastolic blood pressure."
    )
    
    insulin = st.sidebar.number_input(
        'Insulin (mu U/ml)', 
        min_value=0.0, max_value=850.0, 
        value=80.0, step=1.0, 
        format="%.1f",
        help="2-Hour serum insulin."
    )
    
    bmi = st.sidebar.number_input(
        'BMI', 
        min_value=0.0, max_value=70.0, 
        value=30.0, step=0.1, 
        format="%.1f",
        help="Body mass index (weight in kg/(height in m)^2)."
    )
    
    dpf = st.sidebar.number_input(
        'Diabetes Pedigree Function', 
        min_value=0.078, max_value=2.500, 
        value=0.400, step=0.001, 
        format="%.3f",
        help="A genetic score representing the risk of diabetes based on family history."
    )
    
    age = st.sidebar.number_input(
        'Age (years)', 
        min_value=21, max_value=100, 
        value=35, step=1,
        help="Age of the patient."
    )
    
    # Assemble the inputs into a dictionary
    data = {
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Convert the dictionary to a Pandas DataFrame for the model
    features = pd.DataFrame(data, index=[0])
    return features

# Get the user inputs
input_df = user_input_features()

# Display the collected inputs
st.subheader('User Specified Input Parameters')
st.dataframe(input_df, hide_index=True)
st.divider()

# --- Prediction Logic ---

if st.button('🚀 **Predict Diabetes Risk**'):
    # 1. Make Prediction
    # The pipeline handles all preprocessing (scaling/imputation) automatically
    prediction = model_pipeline.predict(input_df)
    
    # 2. Get Confidence Probability
    # Get the probability for each class (0: Non-Diabetic, 1: Diabetic)
    prediction_proba = model_pipeline.predict_proba(input_df)
    
    # 3. Determine the outcome label
    outcome_label = 'Diabetic (Outcome = 1)' if prediction[0] == 1 else 'Non-Diabetic (Outcome = 0)'
    
    # 4. Extract the confidence for the predicted class
    if prediction[0] == 1:
        confidence = prediction_proba[0][1] # Probability of being Diabetic
        st.success('## ✅ Prediction: High Risk!')
    else:
        confidence = prediction_proba[0][0] # Probability of being Non-Diabetic
        st.info('## 💙 Prediction: Low Risk!')
        
    st.write(f"### Result: **{outcome_label}**")
    st.markdown(f"**Confidence**: **{confidence:.2%}**")

    # Optional: Display probability for both classes
    st.text("")
    st.subheader("Probability Breakdown")
    col_0, col_1 = st.columns(2)
    
    # Probability for class 0 (Non-Diabetic)
    col_0.metric(
        label="Non-Diabetic Probability (Outcome 0)",
        value=f"{prediction_proba[0][0]:.2%}",
        delta=None
    )
    
    # Probability for class 1 (Diabetic)
    col_1.metric(
        label="Diabetic Probability (Outcome 1)",
        value=f"{prediction_proba[0][1]:.2%}",
        delta=None
    )