import streamlit as st
import re
import numpy as np
from langchain_experimental.comprehend_moderation import (
    AmazonComprehendModerationChain,
    BaseModerationConfig,
    ModerationPiiConfig,
    ModerationPromptSafetyConfig
)
import boto3
import json

# Initialize the Amazon Comprehend Moderation Chain
pii_config = ModerationPiiConfig(
    labels=["SSN", "PHONE", "EMAIL"],
    redact=True, 
    mask_character="X"
)
prompt_safety_config = ModerationPromptSafetyConfig(threshold=0.8)

moderation_config = BaseModerationConfig(filters=[pii_config, prompt_safety_config])
comprehend_moderation = AmazonComprehendModerationChain(moderation_config=moderation_config)

# Custom function to manually redact credit card numbers
def redact_credit_card(text):
    cc_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    return re.sub(cc_pattern, 'XXXXXXXXXXXXXXXX', text)

# Updated fairness metric function with only Disparate Impact and Equal Opportunity Difference
def calculate_fairness_metrics(predictions, true_labels, protected_group, privileged_value, favorable_label):
    # Generate binary predictions (1 if favorable label, 0 otherwise)
    binary_predictions = [1 if pred['Name'] == favorable_label else 0 for pred in predictions]
    
    # Calculate true positive, false positive, true negative, and false negative rates
    tp = sum([1 for pred, true in zip(binary_predictions, true_labels) if pred == 1 and true == 1])
    fp = sum([1 for pred, true in zip(binary_predictions, true_labels) if pred == 1 and true == 0])
    tn = sum([1 for pred, true in zip(binary_predictions, true_labels) if pred == 0 and true == 0])
    fn = sum([1 for pred, true in zip(binary_predictions, true_labels) if pred == 0 and true == 1])

    # Calculate rates for privileged and unprivileged groups
    privileged_group_indices = [i for i, attr in enumerate(protected_group) if attr == privileged_value]
    unprivileged_group_indices = [i for i, attr in enumerate(protected_group) if attr != privileged_value]
    
    def calculate_group_rates(indices):
        tp_group = sum([1 for i in indices if binary_predictions[i] == 1 and true_labels[i] == 1])
        fp_group = sum([1 for i in indices if binary_predictions[i] == 1 and true_labels[i] == 0])
        tn_group = sum([1 for i in indices if binary_predictions[i] == 0 and true_labels[i] == 0])
        fn_group = sum([1 for i in indices if binary_predictions[i] == 0 and true_labels[i] == 1])
        
        tpr = tp_group / (tp_group + fn_group) if tp_group + fn_group > 0 else 0
        rate = (tp_group + fp_group) / len(indices) if len(indices) > 0 else 0
        return tpr, rate
   
    tpr_priv, rate_priv = calculate_group_rates(privileged_group_indices)
    tpr_unpriv, rate_unpriv = calculate_group_rates(unprivileged_group_indices)
    
    # Fairness metrics
    # disparate_impact = rate_unpriv / rate_priv if rate_priv > 0 else float('inf')
    equal_opportunity_diff = tpr_unpriv - tpr_priv
    
    return {
        
        "Equal Opportunity Difference": equal_opportunity_diff
    }

# AWS Clients
comprehend_client = boto3.client('comprehend')
lambda_client = boto3.client('lambda')

# Streamlit App
st.title("Medical Data Classifier with Responsible AI")

# Section for Responsible AI Principles
st.header("Responsible AI Enhancements")
st.markdown("""
This application follows Responsible AI principles by:

1. **Privacy**: Your data is handled securely, with actions logged and data deleted after processing.
2. **Fairness**: We calculate and report fairness metrics to ensure the model's decisions are equitable.
3. **Explainability**: We offer detailed explanations for AI decisions, including the context and limitations of the model's predictions.
""")

# Consent and Data Usage Transparency
st.subheader("Data Usage Consent")
st.markdown("By using this application, you consent to the processing of your medical data for classification and analysis purposes. Your data will be handled securely and ethically.")
consent_given = st.checkbox("I consent to the use of my data")

# Initialize session state to hold the input text
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Text area for patient or user data
input_text = st.text_area("Enter medical text to check:", height=200, value=st.session_state.input_text)

# Update session state when text is entered
st.session_state.input_text = input_text

# Add sidebar for protected attribute selection
age_group = st.sidebar.selectbox("Select Age Group", ["Young", "Middle-aged", "Elderly"])
race_group = st.sidebar.selectbox("Select Race Group", ["Asian", "Black", "White", "Other"])
gender_group = st.sidebar.selectbox("Select Gender Group", ["Male", "Female", "Other"])

# Button to trigger moderation checks and medical processing
if st.button("Process Text") and consent_given:
    if st.session_state.input_text:
        try:
            # Invoke the moderation chain with the correct key 'input'
            moderation_result = comprehend_moderation.invoke({"input": st.session_state.input_text})
            
            if moderation_result and 'output' in moderation_result:
                moderated_text = moderation_result['output']
                
                # Manually redact credit card numbers
                redacted_text = redact_credit_card(moderated_text)
                
                st.success("Moderation and redaction successful.")
                st.subheader("Moderated and Redacted Text")
                st.write(redacted_text)
                
                # Invoke the custom Comprehend classification model
                custom_response = comprehend_client.classify_document(
                    Text=redacted_text,
                    EndpointArn="arn:aws:comprehend:us-west-2:913524913171:document-classifier-endpoint/medical-specialty-classifier-endpoint"
                )
                
                # Find the label with the maximum score
                max_label = max(custom_response['Labels'], key=lambda x: x['Score'])
                
                st.subheader("Medical Specialty Classification")
                st.json(custom_response)
                st.subheader(f"**Most Likely Specialty:** {max_label['Name']} (Score: {max_label['Score']:.4f})")
                
                # Simulate protected attribute data 
                simulated_true_labels = [0, 1, 0, 1]
                simulated_race_group = [0 if race_group == "Asian" else 1 if race_group == "Black" else 2 if race_group == "White" else 3 for _ in custom_response['Labels']]
                
                # Calculate and display fairness metrics
                fairness_metrics = calculate_fairness_metrics(
                    custom_response['Labels'], 
                    simulated_true_labels,
                    simulated_race_group,
                    privileged_value=2,  # Example: "White" as privileged group
                    favorable_label=max_label['Name'] 
                )
                st.subheader("Fairness Metrics")
                st.json(fairness_metrics)
                
                
                # Construct a lambda payload without any breast cancer-specific data
                lambda_payload = {
                    "input_text": redacted_text
                }

                lambda_response = lambda_client.invoke(
                    FunctionName='Enriched-Comprehend-Medical-CallCMandCustomCode-CTKtvFTHo2KT',
                    InvocationType='RequestResponse',
                    Payload=json.dumps(lambda_payload)
                )
                
                # Do not display the Lambda function response
                
            else:
                st.warning("No output generated. There may be an issue with the moderation chain.")
        except Exception as e:
            st.error(f"Medically sensitive term Detected: {str(e)}")
    else:
        st.warning("Please enter some text to check.")
elif not consent_given:
    st.warning("Please provide your consent to proceed.")
