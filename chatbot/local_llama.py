from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Load .env into environment (if a .env file exists)
load_dotenv()

# Only set environment variables when values are present (avoid None)
for _key in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY"):
    _val = os.getenv(_key)
    if _val is not None:
        os.environ[_key] = _val

# Ensure tracing flag is set to a string value if not provided
if os.getenv('LANGCHAIN_TRACING_V2') is None:
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'



# Define the response schema(s) and output parser before building the prompt
response_schemas = [
    ResponseSchema(name="answer", description="Concise answer to the user's question")
]

output_parser = StructuredOutputParser(response_schemas=response_schemas)

# Include the parser's format instructions in the prompt so the LLM returns
# parseable structured output.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the user queries in a concise manner."),
        # Escape braces so ChatPromptTemplate does not treat JSON style
        # format instructions as template variables.
        ("system", output_parser.get_format_instructions().replace('{', '{{').replace('}', '}}')),
        ("user", "Question: {question}")
    ]
)


# stream lit app
st.title("LangChain Chatbot with Streamlit")
st.write("Ask me anything!")

input_question = st.text_input("Your Question:", "")


llm = Ollama(model='gemma:2b')
chain = prompt | llm | output_parser

if input_question:
    result = chain.invoke({'question': input_question})
    st.write(result['answer'])
