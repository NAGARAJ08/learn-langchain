# data ingestion
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from langchain_community.document_loaders import TextLoader
loader = TextLoader('speech.txt')
text_doc = loader.load()
print(f"Loaded {len(text_doc)} document(s)")
# print(f"Document content length: {len(text_doc[0].page_content)} characters")
# print(f"First 300 characters of document:\n{text_doc[0].page_content[:300]}")


import os
from dotenv import load_dotenv
load_dotenv()
# Only set environment variables when values are present (avoid None)
for _key in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY"):
    _val = os.getenv(_key)
    if _val is not None:
        os.environ[_key] = _val

# Ensure tracing flag is set to a string value if not provided
if os.getenv('LANGCHAIN_TRACING_V2') is None:
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'


#reading from web based load

from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


loader = WebBaseLoader(web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=("post-title","post-content","post-header")
                       )))
web_docs = loader.load()
# print(web_docs)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Attention.pdf")
pdf_docs = loader.load()
# print(pdf_docs)


# Transformations

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents=text_splitter.split_documents(pdf_docs)
print(f"Number of documents: {len(documents)}")
# print(documents[:5])


#vector embedding and vector store
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='gemma:2b')
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
    collection_name='pdf-collection',
    persist_directory='./chroma_db'
)
query = "Who are the authors of attention is all you need?"
# docs = vectorstore.similarity_search(query,k=3)
# print(docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="gemma:2b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain({"query": query})

print("Answer:", result['result'])
print("\nSource Documents:", result['source_documents'])