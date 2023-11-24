import os
import streamlit as st
import requests 
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain import PromptTemplate

#os.environ["OPENAI_API_KEY"] = "abc"
openai_api_key = os.environ.get("OPENAI_API_KEY")
PASSWORD = "sgh"

# Define the URL of your document on GitHub
github_document_url = "https://raw.githubusercontent.com/Jwcbaw34/KM/main/20161027%20MICU%20Medical%20Board%20Paper%20(1%20Nov%202016%20MB%20Meeting).pdf"

# Check if the document is already downloaded; if not, download it
local_document_path = "20161027 MICU Medical Board Paper (1 Nov 2016 MB Meeting).pdf"

if not os.path.exists(local_document_path):
    st.info("Downloading document...")
    response = requests.get(github_document_url)

    if response.status_code == 200:
        with open(local_document_path, "wb") as file:
            file.write(response.content)
        st.success("Document downloaded successfully.")
    else:
        st.error(f"Failed to download document: {response.status_code}")
else:
    st.success("Document already downloaded.")

# Define the directory where you want to store the vector store
VECTORSTORE_PATH = "vectorstore"  # You can customize this path if needed

# If the vector store directory doesn't exist, create it
if not os.path.exists(VECTORSTORE_PATH):
    st.info("Creating the vector store...")

    # Load and process the document
    document = PyMuPDFLoader(local_document_path).load()

    # Process each page in the document (assuming 'document' is a list of pages)
    texts = []
    for page in document:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        page_texts = text_splitter.split_documents([page])
        texts.extend(page_texts)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create the vector store
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(VECTORSTORE_PATH)

    st.success("Vector store created successfully.")
else:
    st.success("Vector store already exists.")
