import os
import importlib
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# PASSWORD PROTECTION FOR UPLOAD
# ----------------------------
APP_PASSWORD = "CommanderKeshav"  # Password for upload access

if "auth" not in st.session_state:
    st.session_state.auth = False

# ----------------------------
# SAFE IMPORT FOR PDF LOADER
# ----------------------------
def get_pypdf_loader():
    """Safely import PyPDFLoader from either langchain_community or langchain."""
    try:
        loader_module = importlib.import_module("langchain_community.document_loaders")
    except ModuleNotFoundError:
        loader_module = importlib.import_module("langchain.document_loaders")
    return loader_module.PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Virtual Teacher", layout="wide")
st.title("📚 Virtual Teacher - Learn from Your PDFs")

# Get API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found! Please set it in your .env file.")
    st.stop()

# Path to store trained vectorstore
VECTORSTORE_PATH = "vectorstore"

# Load existing vectorstore if available
vectorstore = None
if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings(openai_api_key=openai_api_key))

# ----------------------------
# Chat Section (Available to Everyone if trained)
# ----------------------------
if vectorstore:
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
    )
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask me anything about the uploaded documents:")
    if question:
        result = chain({"question": question, "chat_history": st.session_state.history})
        st.session_state.history.append((question, result["answer"]))
        st.write(f"**Answer:** {result['answer']}")

        with st.expander("Chat History"):
            for q, a in st.session_state.history:
                st.write(f"**You:** {q}")
                st.write(f"**Teacher:** {a}")
else:
    st.warning("No trained documents found yet. Upload a PDF to start training.")

# ----------------------------
# Owner Upload Section
# ----------------------------
st.subheader("Owner Access - Upload & Train More PDFs")

if not st.session_state.auth:
    password_input = st.text_input("Enter Upload Password:", type="password")
    if password_input == APP_PASSWORD:
        st.session_state.auth = True
        st.experimental_rerun()
    elif password_input:
        st.error("Incorrect password.")

if st.session_state.auth:
    uploaded_file = st.file_uploader("Upload a PDF to add to training", type="pdf")
    if uploaded_file:
        with st.spinner("Processing your PDF..."):
            temp_path = os.path.join("temp.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            PyPDFLoader = get_pypdf_loader()
            loader = PyPDFLoader(temp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            if vectorstore:
                # Add to existing vectorstore
                vectorstore.add_documents(split_docs)
            else:
                # Create new vectorstore
                vectorstore = FAISS.from_documents(split_docs, embeddings)

            # Save updated vectorstore
            vectorstore.save_local(VECTORSTORE_PATH)

            st.success("Training updated! This document is now part of the knowledge base.")
