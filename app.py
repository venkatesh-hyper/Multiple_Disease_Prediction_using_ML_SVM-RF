import streamlit as st
import numpy as np
import joblib

# Load pre-trained models
svm8020 = joblib.load("svm8020.pkl")
svm6040 = joblib.load("svm6040.pkl")
randfor = joblib.load("RandomForest8020.pkl")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the Streamlit app
st.title("Disease Prediction Based on Symptoms")

# List all possible symptoms (ensure it matches the training order)
all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'chills', 'joint_pain',
    'stomach_pain', 'vomiting', 'fatigue', 'weight_loss', 'anxiety', 'high_fever', 'headache',
    'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'swelling_of_stomach', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'sinus_pressure', 'runny_nose', 'chest_pain', 'weakness_in_limbs', 'pain_during_bowel_movements',
    'neck_pain', 'dizziness', 'cramps', 'obesity', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'excessive_hunger', 'drying_and_tingling_lips', 'slurred_speech', 'muscle_weakness',
    'stiff_neck', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell',
    'bladder_discomfort', 'continuous_feel_of_urine', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'increased_appetite', 'lack_of_concentration', 'visual_disturbances'
]

# Allow user to select multiple symptoms
selected_symptoms = st.multiselect("Select Symptoms", all_symptoms)

# Initialize a binary vector for symptoms
input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

# When the user clicks the button, predict the disease
if st.button("Predict Disease"):
    # Convert input data to numpy array and reshape for prediction
    input_data_np = np.array(input_data).reshape(1, -1)
    
    # Make prediction using the loaded models
    pred_svm8020 = svm8020.predict(input_data_np)
    pred_svm6040 = svm6040.predict(input_data_np)
    pred_rand = randfor.predict(input_data_np)
    
    # Display prediction results
    st.success(f"The SVM_model 1 predicted disease is: {pred_svm8020[0]} based on your symptoms")
    st.success(f"The SVM_model 2 predicted disease is: {pred_svm6040[0]} based on your symptoms")
    st.success(f"The Random_Forest model 3 predicted disease is: {pred_rand[0]} based on your symptoms")
    
    st.warning(f"the prediction is based on machine learning models you may or may not have the disease , so use this prediction as a 1st level analysis so you can futher approach you diagnosis based on prediction")
else:
    st.warning("Please select at least one symptom.")
