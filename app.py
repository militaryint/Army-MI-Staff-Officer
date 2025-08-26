import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# PASSWORD PROTECTION
# ----------------------------
APP_PASSWORD = "CommanderKeshav"  # Your password

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    password_input = st.text_input("Enter App Password:", type="password")
    if password_input == APP_PASSWORD:
        st.session_state.auth = True
        st.experimental_rerun()
    elif password_input:
        st.error("Incorrect password. Try again.")
    st.stop()

# ----------------------------
# SAFE IMPORT FOR PDF LOADER
# ----------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader
except ModuleNotFoundError:
    from langchain.document_loaders import PyPDFLoader

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

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        temp_path = os.path.join("temp.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=vectorstore.as_retriever(),
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        st.success("PDF processed! You can now ask questions.")

        question = st.text_input("Ask me anything about the document:")

        if question:
            result = chain({"question": question, "chat_history": st.session_state.history})
            st.session_state.history.append((question, result["answer"]))
            st.write(f"**Answer:** {result['answer']}")

            with st.expander("Chat History"):
                for q, a in st.session_state.history:
                    st.write(f"**You:** {q}")
                    st.write(f"**Teacher:** {a}")
else:
    st.info("Please upload a PDF to start.")
