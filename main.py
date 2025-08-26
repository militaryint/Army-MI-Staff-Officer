import os, importlib
import streamlit as st
from dotenv import load_dotenv

# ---------- Config ----------
VERSION_TAG = "VT-2025-08-26-1"
APP_PASSWORD = "CommanderKeshav"          # upload gate
VECTORSTORE_DIR = "vectorstore"            # folder, not a single file
PDF_LIST_FILE = "trained_pdfs.txt"
UPLOADS_DIR = "uploads"

# ---------- Boot ----------
load_dotenv()
st.set_page_config(page_title="Virtual Teacher", layout="wide")
st.title("📚 Virtual Teacher — Multi-PDF Knowledge Base")
st.caption(f"Build: {VERSION_TAG} • Main file: {__file__}")

# ---------- Safe imports for LangChain/OpenAI ----------
# (Works with both new + old LangChain integrations)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # new style
except Exception:
    from langchain.chat_models import ChatOpenAI               # fallback
    from langchain.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def get_pypdf_loader():
    """Import PyPDFLoader lazily to avoid ModuleNotFoundError on deploy."""
    try:
        mod = importlib.import_module("langchain_community.document_loaders")
    except ModuleNotFoundError:
        mod = importlib.import_module("langchain.document_loaders")
    return mod.PyPDFLoader

# ---------- API key ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env. Add it like:\nOPENAI_API_KEY=sk-...")
    st.stop()

# ---------- Paths ----------
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------- Load / Save vectorstore ----------
def load_vectorstore(emb):
    if not os.path.exists(VECTORSTORE_DIR):
        return None
    # allow_dangerous_deserialization needed for FAISS pickle metadata
    return FAISS.load_local(VECTORSTORE_DIR, emb, allow_dangerous_deserialization=True)

def save_vectorstore(vs):
    vs.save_local(VECTORSTORE_DIR)

# ---------- Embeddings ----------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ---------- Load existing KB ----------
vectorstore = load_vectorstore(embeddings)

# ---------- Show trained PDF list ----------
st.subheader("📂 Current Knowledge Base")
if os.path.exists(PDF_LIST_FILE):
    with open(PDF_LIST_FILE, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if names:
        for n in names:
            st.write(f"✅ {n}")
    else:
        st.info("No PDFs have been trained yet.")
else:
    st.info("No PDFs have been trained yet.")

# ---------- Chat (available to everyone if trained) ----------
if vectorstore:
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever(),
    )
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("💬 Ask anything about the trained documents:")
    if question:
        result = chain({"question": question, "chat_history": st.session_state.history})
        st.session_state.history.append((question, result["answer"]))
        st.write(f"**Answer:** {result['answer']}")

        with st.expander("📜 Chat History"):
            for q, a in st.session_state.history:
                st.write(f"**You:** {q}")
                st.write(f"**Teacher:** {a}")
else:
    st.warning("⚠️ No documents trained yet. Owner must upload first.")

# ---------- Owner upload (password-gated) ----------
st.subheader("🔐 Owner Access — Upload & Train More PDFs")
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pw = st.text_input("Enter Upload Password:", type="password")
    if pw == APP_PASSWORD:
        st.session_state.auth = True
        st.experimental_rerun()
    elif pw:
        st.error("❌ Incorrect password.")

if st.session_state.auth:
    up = st.file_uploader("Upload a PDF to add to the knowledge base", type="pdf")
    if up is not None:
        # Save original file
        saved_path = os.path.join(UPLOADS_DIR, up.name)
        with open(saved_path, "wb") as f:
            f.write(up.read())

        with st.spinner("Processing your PDF..."):
            PyPDFLoader = get_pypdf_loader()
            loader = PyPDFLoader(saved_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            if vectorstore:
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, embeddings)

            save_vectorstore(vectorstore)

            # Track filename
            with open(PDF_LIST_FILE, "a") as f:
                f.write(up.name + "\n")

            st.success(f"✅ '{up.name}' added to the knowledge base!")
