import requests
import streamlit as st

def get_ollama_response(prompt: str) -> str:
    url = "http://localhost:8000/poem/invoke"
    payload = {"input": {"topic": prompt}}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("output", "No output found sorry")
    else:
        return f"Error: {response.status_code} - {response.text}"
    
    
st.title("Ollama Poem Generator")
user_input = st.text_input("Enter a topic for the poem:")

if user_input:
    result = get_ollama_response(user_input)
    st.subheader("Generated Poem:")
    st.write(result)
    
    