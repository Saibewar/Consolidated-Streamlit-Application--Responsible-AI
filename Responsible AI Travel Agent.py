import streamlit as st
import boto3
import json
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests

 
# AWS client setup using environment variables
try:
    session = boto3.Session()
 
    bedrock_runtime = session.client('bedrock-runtime')
    comprehend = session.client('comprehend')
except (NoCredentialsError, PartialCredentialsError) as e:
    st.error("AWS credentials not found. Please configure your environment correctly.")
 
# Predefined guardrail ID
guardrail_id = 'y334bznjuo9v'
 
# Function to call the AI model and return the necessary output, including any guardrail actions
def call_bedrock_titan_model_with_guardrails(prompt):
    try:
        input_body = {"inputText": prompt}
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-text-lite-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(input_body),
            trace="ENABLED",
            guardrailIdentifier=guardrail_id,
            guardrailVersion="DRAFT"
        )
        output_body = json.loads(response["body"].read().decode())
        action = output_body.get("amazon-bedrock-guardrailAction", "NONE")
        return output_body["results"][0]["outputText"], output_body
    except Exception as e:
        st.error(f"Error invoking model: {e}")
        return None, None
 
# Function to check toxicity using Amazon Comprehend
def check_toxicity_with_comprehend(text):
    try:
        response = comprehend.detect_toxic_content(
            TextSegments=[{"Text": text}],
            LanguageCode='en'
        )
 
        # Remove the "GRAPHIC" label from the response
        for result in response.get("ResultList", []):
            result["Labels"] = [label for label in result["Labels"] if label["Name"] != "GRAPHIC"]
 
        toxicity_detected = any(item['Toxicity'] > 0.5 for item in response['ResultList'])
        return toxicity_detected, response
    except Exception as e:
        st.error(f"Error checking toxicity: {e}")
        return False, None
 
# Function to convert JSON to a table for toxicity details
def json_to_table(toxicity_json):
    table_data = []
    for result in toxicity_json.get("ResultList", []):
        for label in result.get("Labels", []):
            table_data.append({
                "Name": label["Name"],
                "Score": label["Score"]
            })
    if table_data:
        df = pd.DataFrame(table_data)
        return df
    else:
        return None
 
# Function to convert JSON to a table for the trace data
def trace_json_to_table(trace_json):
    trace_data = []
    guardrail_data = trace_json.get("amazon-bedrock-trace", {}).get("guardrail", {}).get("input", {})
    for policy_id, policy_details in guardrail_data.items():
        content_policy = policy_details.get("contentPolicy", {}).get("filters", [])
        word_policy = policy_details.get("wordPolicy", {}).get("managedWordLists", [])
        for item in content_policy:
            trace_data.append({
                "Type": item["type"],
                "Confidence": item.get("confidence", "N/A"),
                "Action": item["action"]
            })
        for item in word_policy:
            trace_data.append({
                "Type": item["type"],
                "Match": item["match"],
                "Action": item["action"]
            })
 
    if trace_data:
        df = pd.DataFrame(trace_data)
        return df
    else:
        return None
 
# Initialize the Google Search API
def search_google(query, api_key, cse_id):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": 5  # Get top 5 results
    }
    response = requests.get(url, params=params)
    return response.json()
 
# Summarize using Hugging Face's summarization pipeline
# Summarize using an Amazon Bedrock model (replacing Hugging Face)
# Summarize using an Amazon Bedrock model (e.g., Anthropic Claude)
# Summarize using an Amazon Bedrock model (e.g., Anthropic Claude)
def summarize_snippets(snippets):
    combined_text = " ".join(snippets)
    prompt = f"Human: Summarize the following text:\n\n{combined_text}\n\nSummary:\nAssistant:"

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',  # Replace with your chosen model
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 200  # Adjust the token limit as needed
        })
    )

    # Parse the response
    output_body = json.loads(response["body"].read().decode())
    summary = output_body["completion"]

    return summary.strip()


 
# Fact-check using AWS Bedrock LLM
# Fact-check using AWS Bedrock LLM (e.g., Anthropic Claude)
def fact_check(statement, summary):
    prompt = f"Human: Please fact-check the following statement: '{statement}' based on the following summary: {summary}\nAssistant:"

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',  # Replace with your chosen model
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 200  # Adjust the token limit as needed
        })
    )

    # Parse the response
    result = json.loads(response["body"].read().decode())
    
    # Extract the fact-checking result
    fact_check_result = result.get('completion', 'No fact-checking result found.')

    return fact_check_result.strip()
 
# Generate a fact-checked response
def generate_fact_check(statement, api_key, cse_id):
    search_results = search_google(statement, api_key, cse_id)
    snippets = [item['snippet'] for item in search_results.get('items', [])]
 
    if not snippets:
        return "No relevant results found for the given statement."
 
    summary = summarize_snippets(snippets)
    sources = "\n".join([f"- [{item['title']}]({item['link']})" for item in search_results.get('items', [])])
 
    fact_check_result = fact_check(statement, summary)
 
    return f"Fact-Check Summary:\n{fact_check_result}\n\nSources:\n{sources}"
 
# Suggested questions about traveling
suggested_questions = [
    "What are the best travel destinations for 2024?",
    "How do I find the cheapest flights?",
    "What should I pack for a two-week vacation?",
    "How can I stay safe while traveling abroad?",
    "What are the top 10 travel tips for first-time travelers?",
    "How do I find eco-friendly travel options?",
    "What are the best ways to book hotels?",
    "How do I plan a budget-friendly trip?",
    "What are the cheapest rates for SeaScanner flights, and can you provide political information about eco-friendly travel options?"
]
 
# Main function to run the Streamlit app
def main():
 
    st.title("Amazon Bedrock Guardrails Demo with Toxicity Detection and Fact-Checking")
 
    selected_question = st.selectbox(
 
        "Choose a question to explore:",
 
        suggested_questions
 
    )
 
    user_input = st.text_input("Or enter your own query", selected_question if selected_question else "")
 
    # Use Streamlit's session state to preserve the response text
 
    if 'response_text' not in st.session_state:
 
        st.session_state.response_text = None
 
    if st.button("Submit Query"):
 
        response_text, output_body = call_bedrock_titan_model_with_guardrails(user_input)
 
        st.session_state.response_text = response_text  # Save the response in session state
 
        st.write("Model Response:", response_text)
 
        if response_text:
 
            # Run the response text through Amazon Comprehend for toxicity detection
 
            toxicity_detected, toxicity_details = check_toxicity_with_comprehend(response_text)
 
            if toxicity_detected:
 
                st.warning("Toxic content detected in the response!")
 
            else:
 
                st.success("No toxic content detected.")
 
            if toxicity_details is not None:
 
                df_toxicity = json_to_table(toxicity_details)
 
                if df_toxicity is not None:
 
                    st.write("Toxicity Details:")
 
                    st.table(df_toxicity)
 
                else:
 
                    st.write("No toxicity labels available.")
 
            else:
 
                st.write("No toxicity details returned by Comprehend.")
 
            df_trace = trace_json_to_table(output_body)
 
            if df_trace is not None:
 
                st.write("Guardrail Trace Details:")
 
                st.table(df_trace)
 
            else:
 
                st.write("No guardrail trace data available.")
 
            guardrail_action = output_body.get("amazon-bedrock-guardrailAction", "NONE")
 
            if guardrail_action and guardrail_action != "NONE":
 
                st.write("Guardrail Action:", guardrail_action)
 
                if isinstance(guardrail_action, dict):
 
                    st.write("Guardrail Intervention:")
 
                    st.json(guardrail_action)
 
                else:
 
                    st.write(f"Unexpected action format: {guardrail_action}")
 
            else:
 
                st.write("No violations detected.")
 
    # Fact-checking feature is triggered by a separate button
 
    if st.session_state.response_text:
 
        if st.button("Fact-Check Response"):
 
            api_key = "AIzaSyBCi3OxYr2UqJ5CScaUrMAXcAtBcGtr8CA"
 
            cse_id = "73536e2a6ab9e4e89"
 
            st.write("Fact-checking the response...")
 
            fact_check_summary = generate_fact_check(st.session_state.response_text, api_key, cse_id)
 
            st.write("Fact-Check Summary:")
 
            st.write(fact_check_summary)
 
 
if __name__ == "__main__":
 
    main()
 
