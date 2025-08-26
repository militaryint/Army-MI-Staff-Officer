import os
import importlib
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
APP_PASSWORD = "CommanderKeshav"
VECTORSTORE_PATH = "vectorstore"
PDF_LIST_FILE = "trained_pdfs.txt"

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------
# PAGE SETTINGS
# ----------------------------
st.set_page_config(page_title="Virtual Teacher", layout="wide")
st.title("📚 Virtual Teacher - Multi-PDF Knowledge Base")

# ----------------------------
# ERROR HANDLER WRAPPER
# ----------------------------
try:
    if not openai_api_key:
        st.error("⚠️ OpenAI API key not found. Please set it in `.env` or Streamlit secrets.")
        st.stop()

    # Safe PDF loader
    def get_pypdf_loader():
        try:
            loader_module = importlib.import_module("langchain_community.document_loaders")
        except ModuleNotFoundError:
            loader_module = importlib.import_module("langchain.document_loaders")
        return loader_module.PyPDFLoader

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS  # ✅ Updated import
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    # Load existing vectorstore
    vectorstore = None
    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            allow_dangerous_deserialization=True
        )

    # Show list of trained PDFs
    st.subheader("📂 Current Knowledge Base")
    if os.path.exists(PDF_LIST_FILE):
        with open(PDF_LIST_FILE, "r") as f:
            pdfs = [line.strip() for line in f if line.strip()]
        if pdfs:
            for pdf in pdfs:
                st.write(f"✅ {pdf}")
        else:
            st.info("No PDFs have been trained yet.")
    else:
        st.info("No PDFs have been trained yet.")

    # Chat section
    if vectorstore:
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=vectorstore.as_retriever(),
        )
        if "history" not in st.session_state:
            st.session_state.history = []

        question = st.text_input("💬 Ask me anything about the documents:")
        if question:
            result = chain({"question": question, "chat_history": st.session_state.history})
            st.session_state.history.append((question, result["answer"]))
            st.write(f"**Answer:** {result['answer']}")

            with st.expander("📜 Chat History"):
                for q, a in st.session_state.history:
                    st.write(f"**You:** {q}")
                    st.write(f"**Teacher:** {a}")
    else:
        st.warning("⚠️ No documents trained yet. Owner needs to upload a PDF.")

    # Owner upload section
    st.subheader("🔐 Owner Access – Upload & Train More PDFs")

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        password_input = st.text_input("Enter Upload Password:", type="password")
        if password_input == APP_PASSWORD:
            st.session_state.auth = True
            st.experimental_rerun()
        elif password_input:
            st.error("❌ Incorrect password.")

    if st.session_state.auth:
        uploaded_file = st.file_uploader("Upload a PDF to add to the knowledge base", type="pdf")
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
                    vectorstore.add_documents(split_docs)
                else:
                    vectorstore = FAISS.from_documents(split_docs, embeddings)

                vectorstore.save_local(VECTORSTORE_PATH)

                # Save PDF name to list
                with open(PDF_LIST_FILE, "a") as f:
                    f.write(f"{uploaded_file.name}\n")

                st.success(f"✅ '{uploaded_file.name}' added to the knowledge base!")

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
