import streamlit as st
import boto3
import json
import hmac
import hashlib
import base64
from botocore.exceptions import ClientError
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd

# AWS configurations
REGION_NAME = "us-west-2"
USER_POOL_ID = "us-west-2_VjeXFctoA"
CLIENT_ID = "fmoimskr32dcu1gjc2e8a1ou0"
# CLIENT_SECRET = "d4clgkrrso0j6farp1op6u0thg4ff1qbuo2v7qkic4clp0iach"  # Replace with your actual client secret
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
GUARDRAIL_IDS = {
    "Doctor": "p3lgqv4onhcc",
    "Admin": "yi3w386mp288",
    "Nurse": "wdc8660lsw70"
}
MAX_TOKENS = 1000
GUARDRAIL_VERSION = "DRAFT"
S3_BUCKET_NAME = "healthcare-guardrail-data"

# Initialize AWS clients
session = boto3.Session(region_name=REGION_NAME)
cognito_client = session.client('cognito-idp')
bedrock_client = session.client('bedrock-runtime')
s3_client = session.client('s3')
comprehend_client = session.client('comprehend')

"""def get_secret_hash(username):
    msg = username + CLIENT_ID
    dig = hmac.new(str(CLIENT_SECRET).encode('utf-8'), msg=str(msg).encode('utf-8'), digestmod=hashlib.sha256).digest()
    return base64.b64encode(dig).decode()"""

def authenticate_user(username, password):
    try:
        response = cognito_client.admin_initiate_auth(
            UserPoolId=USER_POOL_ID,
            ClientId=CLIENT_ID,
            AuthFlow='ADMIN_NO_SRP_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password
                # 'SECRET_HASH': get_secret_hash(username)
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

def load_data_from_s3():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key='Unique_Healthcare_Data_200.txt')
        data = response['Body'].read().decode('utf-8')
        return data
    except ClientError as e:
        st.error(f"Failed to load data from S3: {e}")
        return None

def generate_analysis(prompt, guardrail_id, data):
    context = f"Based on the following healthcare data:\n\n{data}\n\n"
    full_prompt = context + prompt

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
                    }
                ]
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
        return None

def perform_sentiment_analysis(text):
    try:
        sentiment_response = comprehend_client.detect_sentiment(
            LanguageCode='en',
            Text=text
        )
        return sentiment_response
    except ClientError as e:
        st.error(f"Error performing sentiment analysis: {e}")
        return None

def convert_to_segments(text, max_bytes=1000):
    segments = []
    current_segment = ""

    for sentence in sent_tokenize(text):
        if len((current_segment + " " + sentence).encode('utf-8')) <= max_bytes:
            current_segment += " " + sentence
        else:
            segments.append({"Text": current_segment.strip()})
            current_segment = sentence

    if current_segment:
        segments.append({"Text": current_segment.strip()})

    return segments

def detect_toxicity(segmented_text):
    toxicity_results = []
    
    for segment in segmented_text:
        try:
            toxic = comprehend_client.detect_toxic_content(
                TextSegments=[segment],
                LanguageCode='en'
            )
            toxicity_results.extend(toxic['ResultList'])
        except ClientError as e:
            st.error(f"Error detecting toxicity: {e}")
            return None
    
    return {"ResultList": toxicity_results}

def main():
    st.title("Healthcare Application")

    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = False
        st.session_state.new_password_required = False

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
                st.session_state.auth_status = True
                st.rerun()
            except ClientError as e:
                st.error(f"Failed to update password: {e}")

    if st.session_state.auth_status:
        st.write(f"Welcome, {st.session_state.username}!")
        user_group = get_user_group(st.session_state.username)
       
        if user_group:
            st.write(f"You are in the {user_group} group.")
            guardrail_id = GUARDRAIL_IDS.get(user_group)
           
            if guardrail_id:
                user_prompt = st.text_input("Please enter your query:")

                if st.button("Generate Answer"):
                    if user_prompt:
                        data = load_data_from_s3()
                        if data:
                            response_from_llm = generate_analysis(user_prompt, guardrail_id, data)
                            if response_from_llm:
                                st.write("##### Response")
                                st.write(response_from_llm)

                                # Perform sentiment analysis
                                sentiment_response = perform_sentiment_analysis(response_from_llm)
                                if sentiment_response:
                                    st.write("##### Sentiment Analysis")
                                    st.markdown(f"""
                                    - **Sentiment:** {sentiment_response["Sentiment"]}
                                    - **Sentiment Score:**
                                        - Positive: {sentiment_response['SentimentScore']['Positive'] * 100:.2f}%
                                        - Negative: {sentiment_response['SentimentScore']['Negative'] * 100:.2f}%
                                        - Neutral: {sentiment_response['SentimentScore']['Neutral'] * 100:.2f}%
                                        - Mixed: {sentiment_response['SentimentScore']['Mixed'] * 100:.2f}%
                                    """)

                                
                                st.write("##### Toxicity Detection")
                                segmented_text = convert_to_segments(response_from_llm)
                                toxic_content = detect_toxicity(segmented_text)
                                results = toxic_content['ResultList']

                                toxicity_list = []
                                labels_list = []

                                for result in results:
                                    toxicity_list.append(result['Toxicity'])
                                    labels_list.append({label['Name']: label['Score'] for label in result['Labels']})

                                df1 = pd.DataFrame(labels_list)
                                df1['TOXICITY'] = toxicity_list

                                text_list = []
                                for segment in segmented_text:
                                    text_list.append(segment['Text'])

                                df1['SEGMENT'] = text_list

                                new_column_order = ['SEGMENT', 'PROFANITY', 'HATE_SPEECH', 'INSULT', 'GRAPHIC', 'HARASSMENT_OR_ABUSE', 'SEXUAL', 'VIOLENCE_OR_THREAT', 'TOXICITY']
                                df1 = df1.reindex(columns=new_column_order, fill_value='N/A')

                                selected_columns = ['SEGMENT', 'PROFANITY', 'HATE_SPEECH', 'INSULT', 'VIOLENCE_OR_THREAT', 'TOXICITY']
                                df2 = df1[selected_columns]

                                st.write(df2)
                    else:
                        st.warning("Please enter a question for analysis.")
            else:
                st.error(f"No guardrail found for the {user_group} group.")
        else:
            st.error("Failed to retrieve user group.")
 
        if st.button("Logout"):
            st.session_state.auth_status = False
            st.rerun()

if __name__ == "__main__":
    main()
