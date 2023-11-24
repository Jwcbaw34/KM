import os
import streamlit as st
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

st.write('Hello world! testing 1 2 3 ')
