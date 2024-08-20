import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError

# AWS configurations
REGION_NAME = "us-west-2"
USER_POOL_ID = "us-west-2_vuFM4JziF"
CLIENT_ID = "5imlgrjejh2ff0ifh9i1qmsn5j"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
GUARDRAIL_IDS = {
    "Employee": "8ukube0d58o5",
    "Accounts_Team": "lhxdia1ch15i",
    "Portfolio_Manager": "bu44m3fox698",
    "Admin": "yi3w386mp288"
}

MAX_TOKENS = 1000
GUARDRAIL_VERSION = "DRAFT"
S3_BUCKET_NAME = "stock-market-data-poc"

# Initialize AWS clients
session = boto3.Session(region_name=REGION_NAME)
cognito_client = session.client('cognito-idp')
bedrock_client = session.client('bedrock-runtime')
s3_client = session.client('s3')
comprehend_client = session.client('comprehend')

def authenticate_user(username, password):
    try:
        response = cognito_client.admin_initiate_auth(
            UserPoolId=USER_POOL_ID,
            ClientId=CLIENT_ID,
            AuthFlow='ADMIN_NO_SRP_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password
            }
        )
        return response
    except ClientError as e:
        st.error(f"Authentication failed: {e}")
        return None

def get_user_group(username):
    try:
        response = cognito_client.admin_list_groups_for_user(
            Username=username,
            UserPoolId=USER_POOL_ID
        )
        groups = [group['GroupName'] for group in response['Groups']]
        return groups[0] if groups else None
    except ClientError as e:
        st.error(f"Failed to get user group: {e}")
        return None

def get_s3_file_content(file_name):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        file_content = response['Body'].read().decode('utf-8')
        return file_content
    except ClientError as e:
        st.error(f"Failed to retrieve file from S3: {e}")
        return None

def generate_analysis(prompt, context, guardrail_id):
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnalysis:"
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    }

    try:
        response = bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            guardrailIdentifier=guardrail_id,
            guardrailVersion=GUARDRAIL_VERSION
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except ClientError as e:
        st.error(f"Failed to generate analysis: {e}")
        if 'Error' in e.response and 'Message' in e.response['Error']:
            st.error(f"Error details: {e.response['Error']['Message']}")
        return None

def detect_pii_entities(text):
    try:
        response = comprehend_client.detect_pii_entities(
            Text=text,
            LanguageCode='en'
        )
        return response['Entities']
    except ClientError as e:
        st.error(f"Failed to detect PII entities: {e}")
        return None

def analyze_sentiment(text):
    try:
        response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )
        return response['Sentiment'], response['SentimentScore']
    except ClientError as e:
        st.error(f"Failed to analyze sentiment: {e}")
        return None, None

def main():
    st.title("Investment Analysis App")

    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = False

    if not st.session_state.auth_status:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            auth_response = authenticate_user(username, password)
            if auth_response:
                if auth_response.get('ChallengeName') == 'NEW_PASSWORD_REQUIRED':
                    st.session_state.new_password_required = True
                    st.session_state.auth_response = auth_response
                    st.session_state.username = username
                else:
                    st.session_state.auth_status = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()

        if 'new_password_required' in st.session_state and st.session_state.new_password_required:
            st.subheader("New Password Required")
            new_password = st.text_input("New Password", type="password")
            if st.button("Set New Password"):
                try:
                    challenge_response = cognito_client.respond_to_auth_challenge(
                        ClientId=CLIENT_ID,
                        ChallengeName='NEW_PASSWORD_REQUIRED',
                        Session=st.session_state.auth_response['Session'],
                        ChallengeResponses={
                            'USERNAME': st.session_state.username,
                            'NEW_PASSWORD': new_password
                        }
                    )
                    st.success("Password updated successfully. Please log in with your new password.")
                    st.session_state.new_password_required = False
                except ClientError as e:
                    st.error(f"Failed to update password: {e}")

    if st.session_state.auth_status:
        st.write(f"Welcome, {st.session_state.username}!")
        user_group = get_user_group(st.session_state.username)
        if user_group:
            st.write(f"You are in the {user_group} group.")
            guardrail_id = GUARDRAIL_IDS.get(user_group)
            if guardrail_id:
                file_content = get_s3_file_content("Customer_Data.txt")
                if file_content:
                    st.subheader("Custom Question")
                    user_prompt = st.text_area("Enter your custom analysis question:")
                    if st.button("Generate Custom Analysis"):
                        if user_prompt:
                            st.info("Generating analysis... Please wait.")
                            analysis = generate_analysis(user_prompt, file_content, guardrail_id)
                            if analysis:
                                st.write("Analysis Result:")
                                st.write(analysis)
                               
                                # PII Detection (Updated to exclude scores)
                                st.subheader("PII Detection")
                                pii_entities = detect_pii_entities(analysis)
                                if pii_entities:
                                    st.write("Detected PII Entities in the analysis:")
                                    for entity in pii_entities:
                                        detected_text = analysis[entity['BeginOffset']:entity['EndOffset']]
                                        st.write(f"- Type: {entity['Type']}, Text: '{detected_text}'")
                                else:
                                    st.write("No PII entities detected in the analysis.")
                               
                                # Sentiment Analysis
                                st.subheader("Sentiment Analysis")
                                sentiment, sentiment_score = analyze_sentiment(analysis)
                                if sentiment and sentiment_score:
                                    st.write(f"Overall Sentiment of the analysis: {sentiment}")
                                    st.write("Sentiment Scores:")
                                    for key, value in sentiment_score.items():
                                        st.write(f"- {key}: {value:.2f}")
                                else:
                                    st.write("Failed to analyze sentiment of the analysis.")
                            else:
                                st.error("Failed to generate analysis. Please check the error messages above for more details.")
                        else:
                            st.warning("Please enter a question for analysis.")
                else:
                    st.error("Failed to retrieve file content. Please try again later.")
            else:
                st.error(f"No guardrail found for the {user_group} group.")
        else:
            st.error("Failed to retrieve user group.")

        if st.button("Logout"):
            st.session_state.auth_status = False
            st.rerun()

if __name__ == "__main__":
    main()
