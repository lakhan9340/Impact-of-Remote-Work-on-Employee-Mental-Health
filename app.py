import joblib
import streamlit as st
import numpy as np
# Load model and individual encoders
model = joblib.load('best_model.joblib')
label_encoders = joblib.load('label_encoders.joblib') # This is a dictionary of encoders
scaler = joblib.load('scaler.joblib')

# Title and inputs
st.title('Mental Health Prediction')
st.write('Enter the details to predict mental health condition')

# Inputs for each feature, accessing the encoder for each column individually
gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
age = st.number_input('Age', min_value=18, max_value=65, step=1)
job_role = st.selectbox('Job Role', label_encoders['Job_Role'].classes_)
industry = st.selectbox('Industry', label_encoders['Industry'].classes_)
year_of_experience = st.number_input('Year of Experience', min_value=0, max_value=40, step=1)
work_location = st.selectbox('Work Location', label_encoders['Work_Location'].classes_)
hours_worked_per_week = st.number_input('Hours Worked Per Week', min_value=0, max_value=60, step=1)
number_of_virtual_meetings = st.number_input('Number of Virtual Meetings', min_value=0, max_value=15, step=1)
work_life_balance_rating = st.number_input('Work Life Balance Rating', min_value=1, max_value=5, step=1)
stress_level = st.selectbox('Stress Level', label_encoders['Stress_Level'].classes_)
productivity_change = st.selectbox('Productivity Change', label_encoders['Productivity_Change'].classes_)
sleep_quality = st.selectbox('Sleep Quality', label_encoders['Sleep_Quality'].classes_)
social_isolation_rating = st.number_input('Social Isolation Rating', min_value=1, max_value=5, step=1)
satisfaction_with_remote_work = st.selectbox('Satisfaction with Remote Work', label_encoders['Satisfaction_with_Remote_Work'].classes_)
company_support_for_remote_work = st.number_input('Company Support for Remote Work', min_value=1, max_value=5, step=1)
access_to_mental_health_resources = st.selectbox('Access to Mental Health Resources', label_encoders['Access_to_Mental_Health_Resources'].classes_)
physical_activity = st.selectbox('Physical Activity', label_encoders['Physical_Activity'].classes_)
region = st.selectbox('Region', label_encoders['Region'].classes_)

# Encode categorical inputs
encoded_inputs = np.array([
    label_encoders['Gender'].transform([gender])[0],
    label_encoders['Job_Role'].transform([job_role])[0],
    label_encoders['Industry'].transform([industry])[0],
    label_encoders['Work_Location'].transform([work_location])[0],
    label_encoders['Stress_Level'].transform([stress_level])[0],
    label_encoders['Productivity_Change'].transform([productivity_change])[0],
    label_encoders['Sleep_Quality'].transform([sleep_quality])[0],
    label_encoders['Satisfaction_with_Remote_Work'].transform([satisfaction_with_remote_work])[0],
    label_encoders['Access_to_Mental_Health_Resources'].transform([access_to_mental_health_resources])[0],
    label_encoders['Physical_Activity'].transform([physical_activity])[0],
    label_encoders['Region'].transform([region])[0]
])

# Numerical inputs (no encoding, just numerical features)
numerical_inputs = np.array([
    age,
    year_of_experience,
    hours_worked_per_week,
    number_of_virtual_meetings,
    work_life_balance_rating,
    social_isolation_rating,
    company_support_for_remote_work
]).reshape(1, -1)

# Combine encoded categorical and numerical inputs into a single array
combined_inputs = np.hstack((encoded_inputs, numerical_inputs.flatten())).reshape(1, -1)

# Scale the combined input features
scaled_inputs = scaler.transform(combined_inputs)

# Predict mental health condition
if st.button("Predict Mental Health Condition"):
    prediction = model.predict(scaled_inputs)
    st.write(f"Mental Health Condition: {prediction[0]}")


