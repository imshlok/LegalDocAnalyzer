# Enhanced Financial Document AI Assistant using LLaMA (Ollama), LangChain, FAISS, Streamlit

import os
import streamlit as st
import tempfile
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# UI Configuration
st.set_page_config(page_title="📊 Financial Document Analyzer", layout="wide")
st.title("📑 AI-Powered Financial Document Analyzer")
st.markdown("""
Welcome to the Financial Document Analyzer. Upload any financial document — bank statements, insurance policies,
company annual reports, index methodologies, etc. — and get AI-powered answers and summaries.
""")

# LLM Initialization
llm = OllamaLLM(model="llama3.2")

# File Upload
uploaded_file = st.file_uploader("📎 Upload a financial document (PDF only)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # Document Loading & Splitting
    loader = PyPDFLoader(tmp_pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)

    # Vector Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

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
        if query:
            with st.spinner("Generating answer..."):
                answer = qa.run(query)
            st.success("Answer")
            st.write(answer)

    with tab2:
        st.subheader("📋 AI-Generated Document Summary")
        if st.button("Summarize Document"):
            full_text = "\n".join([c.page_content for c in chunks])[:3500]
            summary_prompt = (
                "Please analyze and summarize the financial document below in 5-7 bullet points. "
                "Highlight key figures, trends, metrics, clauses, or financial insights.\n\n" + full_text
            )
            with st.spinner("Summarizing document..."):
                summary = llm.invoke(summary_prompt)
            st.success("📌 Summary")
            st.markdown(summary)

    # Cleanup
    try:
        os.remove(tmp_pdf_path)
    except:
        pass

    st.markdown("---")
    st.markdown("**ℹ️ Tip:** You can upload various types of financial documents — earnings reports, stock summaries, IPO filings, etc.")

else:
    st.info("👈 Please upload a financial document to begin analysis.")
