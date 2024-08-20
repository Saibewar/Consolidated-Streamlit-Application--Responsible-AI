import json
import time
import boto3
import streamlit as st
from botocore.exceptions import ClientError

# AWS Configuration
region_name = 'us-east-1'
s3_bucket_name = 'underwriting-document-bucket-101'
step_function_arn = "arn:aws:states:us-east-1:913524913171:stateMachine:UnderwritingValidationStateMachine"

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=region_name)
stepfunctions_client = boto3.client('stepfunctions', region_name=region_name)

# Function to list files in S3 bucket
def list_files_in_s3(bucket_name):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            return [content['Key'] for content in response['Contents']]
        else:
            return []
    except ClientError as e:
        st.error(f"An error occurred while listing files in S3: {e}")
        return []

# Function to upload file to S3
def upload_file_to_s3(file, bucket_name, object_name=None):
    try:
        if object_name is None:
            object_name = file.name
        s3_client.upload_fileobj(file, bucket_name, object_name)
        st.success(f"File uploaded successfully to {bucket_name}/{object_name}")
        return object_name
    except ClientError as e:
        st.error(f"An error occurred while uploading to S3: {e}")
        return None

# Function to start a Step Function execution
def start_step_function(state_machine_arn, input_data):
    try:
        response = stepfunctions_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps(input_data)
        )
        return response['executionArn']
    except ClientError as e:
        st.error(f"An error occurred while starting the Step Function: {e}")
        return None

# Function to get the result of a Step Function execution
def get_step_function_result(execution_arn):
    while True:
        try:
            response = stepfunctions_client.describe_execution(
                executionArn=execution_arn
            )
            status = response['status']
            if status == 'SUCCEEDED':
                return json.loads(response['output'])
            elif status in ['FAILED', 'TIMED_OUT', 'ABORTED']:
                st.error(f"Step Function execution failed with status: {status}")
                return None
            time.sleep(2)
        except ClientError as e:
            st.error(f"An error occurred while retrieving execution result: {e}")
            return None

# Function to parse and display the decision and rationale
def display_decision_rationale(content):
    if content:
        text = content[0]['text']
        decision_start = text.find('<decision>') + len('<decision>')
        decision_end = text.find('</decision>')
        rationale_start = text.find('<rationale>') + len('<rationale>')
        rationale_end = text.find('</rationale>')

        decision = text[decision_start:decision_end].strip()
        rationale = text[rationale_start:rationale_end].strip()

        st.markdown("### Decision")
        st.write(decision)

        st.markdown("### Rationale")
        st.write(rationale)

# Main Streamlit application
def main():
    st.title("Insurance Underwriting Through License JPEG")
    
    st.markdown("## Select or Upload an Image for Underwriting Analysis")
    
    with st.expander("Select an Image"):
        s3_files = list_files_in_s3(s3_bucket_name)
        selected_file = st.selectbox("Select an image from S3", s3_files)

    with st.expander("Upload a New Image"):
        uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            object_name = upload_file_to_s3(uploaded_file, s3_bucket_name)
            selected_file = object_name

    if selected_file:
        st.markdown(f"### Selected Image: `{selected_file}`")
        
        if st.button("Start Underwriting Process"):
            input_data = {
                "detail": {
                    "s3_bucket": s3_bucket_name,
                    "s3_key": selected_file
                }
            }
            execution_arn = start_step_function(step_function_arn, input_data)
            
            if execution_arn:
                st.info("Step Function started, waiting for result...")
                result = get_step_function_result(execution_arn)
                
                if result:
                    st.markdown("### Underwriting Result:")
                    #st.json(result)
                    display_decision_rationale(result["Body"]["content"])

# Running the main function
if __name__ == "__main__":
    main()
