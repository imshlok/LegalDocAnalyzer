# Enhanced Financial Document AI Assistant using LLaMA (Ollama), LangChain, FAISS, Streamlit

import os
import streamlit as st
import tempfile
import hashlib
import base64
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import static_css

# UI Configuration
st.set_page_config(page_title="📊 Financial Document Analyzer", layout="wide")

# Dark mode toggle
is_dark_mode = st.sidebar.toggle("🌗 Dark Mode", value=False)

# Custom CSS Styling
background_style = static_css.select_theme(is_dark_mode)

st.markdown(f"""<style>{background_style}</style>""", unsafe_allow_html=True)

st.title("📑 AI-Powered Financial Document Analyzer")
st.markdown("""
Welcome to the Financial Document Analyzer. Upload any financial document — bank statements, insurance policies,
company annual reports, index methodologies, etc. — and get AI-powered answers and summaries.
""")

# LLM Initialization
llm = OllamaLLM(model="llama3.2")

# Cache embedding model
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

embeddings = load_embeddings()

# File Upload
uploaded_file = st.file_uploader("📎 Upload a financial document (PDF only)", type="pdf")

# Utility: Create hash for caching
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Prepare cache directory
cache_dir = "vector_cache"
os.makedirs(cache_dir, exist_ok=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    file_hash = get_file_hash(tmp_pdf_path)
    db_path = os.path.join(cache_dir, file_hash)

    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(tmp_pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(pages)

        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(db_path)

    # QA Chain Setup
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        chain_type="stuff"
    )

    # Tabs for User Interaction
    tab1, tab2 = st.tabs(["🔍 Ask a Question", "🧠 Get a Summary"])

    with tab1:
        st.subheader("🔎 Ask AI about this document")
        query = st.text_input("Type your financial question", placeholder="e.g. What are the account balances? What is the expense breakdown?")

        sample_questions = [
            "What is the total revenue in the report?",
            "Summarize the investment portfolio.",
            "What are the key expenses mentioned?",
            "Is there any profit/loss trend shown?",
            "Explain the policy clauses."
        ]

        col_buttons = st.columns(len(sample_questions))
        for idx, (col, question) in enumerate(zip(col_buttons, sample_questions)):
            if col.button(question, key=f"sample_{idx}"):
                query = question

        if query:
            with st.spinner("Generating answer..."):
                answer = qa.run(query)
            st.success("Answer")
            st.write(answer)

    with tab2:
        st.subheader("📋 AI-Generated Document Summary")
        if st.button("Summarize Document"):
            loader = PyPDFLoader(tmp_pdf_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(pages)
            full_text = "\n".join([c.page_content for c in chunks])[:3500]
            summary_prompt = (
                "Please analyze and summarize the financial document below in 5-7 bullet points. "
                "Highlight key figures, trends, metrics, clauses, or financial insights.\n\n" + full_text
            )
            with st.spinner("Summarizing document..."):
                summary = llm.invoke(summary_prompt)
            st.success("📌 Summary")
            st.markdown(summary)

        with st.expander("📄 Preview Uploaded PDF"):
            with open(tmp_pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    # Cleanup
    try:
        os.remove(tmp_pdf_path)
    except:
        pass

    st.markdown("---")
    st.markdown("**ℹ️ Tip:** You can upload various types of financial documents — earnings reports, stock summaries, IPO filings, etc.")

else:
    st.markdown("<div class='file-upload-prompt'>👆 Please upload a financial document to begin analysis.</div>", unsafe_allow_html=True)
