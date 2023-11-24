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

# Call set_page_config here at the top-level scope
st.set_page_config(page_title="Knowledge Mgmt Chatbot :bulb:", page_icon=":bulb:")

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

template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

def handler_verify():
    input_password = st.session_state.password_input
    if input_password == PASSWORD:
        st.session_state.password_flag = True
    else:
        st.write("Incorrect Password")

def initialize_app():
    st.text_input(
        label = "Enter Password",
        key = "password_input",
        type = "password",
        on_change = handler_verify,
        placeholder = "Enter Password",
        )


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

def get_qasource_chain():
    qasource_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1024),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
        },
        return_source_documents=True,
        verbose=True
    )
    return qasource_chain

def handle_userinput(user_question):
    response = st.session_state.qasource_chain({"query": user_question})

    # Add user's question and bot's response to the chat history
    st.session_state.chat_history.append(('user', user_question))
    st.session_state.chat_history.append(('bot', response['result']))

    # Display the entire chat history
    for sender, message in st.session_state.chat_history:
        if sender == 'user':
            st.write(
                user_template.replace("{{MSG}}", message),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message),
                unsafe_allow_html=True
            )

    # Display source documents if available
    source_documents = response.get("source_documents", [])
    for i, doc in enumerate(source_documents):
        st.write(f"Document {i+1} Metadata:")
        st.write(f"Source: {doc.metadata['source']}")
        st.write("------")

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "password_flag" not in st.session_state:
        st.session_state.password_flag = False

    #st.set_page_config(page_title="Knowledge Mgmt Chatbot :bulb:", page_icon=":bulb:")
    st.write(css, unsafe_allow_html=True)

    if st.session_state.password_flag:
        if "qasource_chain" not in st.session_state:
            st.session_state.qasource_chain = get_qasource_chain()

        st.header("Knowledge Mgmt Chatbot :bulb:")
        user_question = st.text_input("Ask about innovation/improvement projects:")

        if user_question:
            handle_userinput(user_question)

    else:
        initialize_app()

if __name__ == "__main__":
    main()
