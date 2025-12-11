from fastapi import FastAPI

from langchain.prompts import ChatPromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
# routest are used to interact with dff models
import uvicorn
import os


from langchain_community.llms import Ollama

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


app = FastAPI(
    title='Langchain server',
    version="1.0",
    description = "A simple API server"
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path='/ollama',
# )
# model = ChatOpenAI()
llm = Ollama(model='gemma:2b')

# prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} around 5 lines")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} around 5 lines")

add_routes(
    app,
    prompt2|llm,
    path='/poem'
)


if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=8000,log_level='debug')