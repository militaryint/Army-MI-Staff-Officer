import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
SECRET_KEY = st.secrets["app_key"]
DB_PATH = "faiss_store"

st.set_page_config(page_title="Secure PDF Chatbot", layout="centered")
st.title("🔒 Secure Hugging Face Style PDF Chatbot")

# ---------------- SECRET KEY CHECK ----------------
user_key = st.text_input("Enter Access Key (leave blank for chat only)", type="password")
is_admin = (user_key == SECRET_KEY)

# ---------------- TRAIN PDF SECTION (ADMIN ONLY) ----------------
if is_admin:
    st.subheader("📘 Upload & Train a PDF")
    uploaded_file = st.file_uploader("Upload a PDF to train", type=["pdf"])
    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if os.path.exists(DB_PATH):
                vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
                vectorstore.add_documents(docs)
            else:
                vectorstore = FAISS.from_documents(docs, embeddings)

            vectorstore.save_local(DB_PATH)
            st.success(f"✅ PDF '{uploaded_file.name}' trained and stored in FAISS database!")

            os.remove(temp_path)

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# ---------------- CHAT SECTION ----------------
st.subheader("💬 Chat with Trained PDFs")

if not os.path.exists(DB_PATH):
    st.warning("⚠️ No trained database found! Please upload a PDF (admin only).")
    st.stop()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

llm_pipeline = load_llm()

def ask_llm(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm_pipeline(prompt, max_length=256, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"]

query = st.text_input("💡 Ask a question:")
if query:
    results = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in results])
    answer = ask_llm(query, context)

    st.markdown("**🤖 Answer:**")
    st.write(answer)

    with st.expander("📄 Retrieved context"):
        for i, res in enumerate(results, start=1):
            st.write(f"**{i}.** {res.page_content}")
