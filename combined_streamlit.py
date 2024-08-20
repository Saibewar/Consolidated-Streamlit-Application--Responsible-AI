import streamlit as st
import os

# List of app files
app_files = {
    "Medical Diagnosis Classifier": "medical_classifier.py",
    "Healthcare Application": "healthcare_guardrails.py",
    "Underwriting Auto Insurance":"UnderwritingManualLambdaStreamlit.py",
    "Investment Analysis": "Finance_Guardrails.py",
    "Travel Agent Application": "Responsible AI Travel Agent.py"
}

# Main UI
st.sidebar.title("Responsible AI POC")
app_choice = st.sidebar.radio("Choose an app", list(app_files.keys()))

# Display the selected app
if app_choice:
    file_path = app_files[app_choice]
    
    # Read and execute the selected app
    with open(file_path, 'r') as f:
        code = f.read()
        exec(code, globals())
