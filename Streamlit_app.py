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

os.environ["OPENAI_API_KEY"] = "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"
PASSWORD = "sgh"

MAIN_DIR = "/content/KM_POC"
VECTORSTORE_PATH = os.path.join(MAIN_DIR, "vectorstore")

template = \"\"\"
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
\"\"\"

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

# If vectorstore doesn't exist, create it
if not os.path.exists(VECTORSTORE_PATH):
    document_files = [os.path.join(MAIN_DIR, "Gary Ong files", path) for path in os.listdir(os.path.join(MAIN_DIR, "Gary Ong files"))]
    documents = []
    for doc_file in document_files:
        documents.extend(PyMuPDFLoader(doc_file).load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(VECTORSTORE_PATH)

# Load the created database
docsearch = FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings())


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

    st.set_page_config(page_title="Knowledge Mgmt Chatbot :bulb:", page_icon=":bulb:")
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
