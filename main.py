# Enhanced Financial Document AI Assistant using LLaMA (Ollama), LangChain, FAISS, Streamlit

import os
import streamlit as st
import tempfile
import hashlib
import base64
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import static_css
import yfinance as yf
import requests

NEWS_API_KEY = os.getenv("NEWSAPI_KEY", "4b9740904b514af3a1071c7a1250133c")

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

# === 📈 Market Insights Section (Independent in Sidebar) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Market Sentiment & News")

selected_ticker = st.sidebar.text_input("Stock symbol or company name", value="TCS.NS", key="sidebar_ticker")
period = st.sidebar.selectbox("Stock data period", ["1mo", "3mo", "6mo", "1y"], index=2, key="sidebar_period")

# Cache resolved tickers across session
if "ticker_cache" not in st.session_state:
    st.session_state.ticker_cache = {}

if st.sidebar.button("Analyze Market", key="sidebar_analyze"):
    import time

    # === 🧊 Cooldown Logic ===
    if "last_api_call" not in st.session_state:
        st.session_state.last_api_call = 0

    cooldown_seconds = 10  # You can adjust this if needed

    now = time.time()
    elapsed = now - st.session_state.last_api_call

    if elapsed < cooldown_seconds:
        st.warning(f"⏳ Please wait {int(cooldown_seconds - elapsed)} more seconds before making another request.")
        st.stop()

    # ✅ Update last API call time
    st.session_state.last_api_call = now

    st.markdown("## 📈 Market Sentiment & News Analysis")

    try:
        query = selected_ticker.strip()
        print(f"User searched for: {query}")
        resolved_ticker = query

        # 🔁 Check cache first
        if query in st.session_state.ticker_cache:
            resolved_ticker = st.session_state.ticker_cache[query]
            st.info(f"ℹ️ Using cached ticker: `{resolved_ticker}`")
        elif "." not in query:  # Heuristic: probably a company name
            st.info(f"🔍 Searching ticker for: {query}")
            search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
            time.sleep(1.2)  # Reduce Yahoo rate limit pressure

            search_response = requests.get(search_url)
            print(f"data found: {search_response}")
            if search_response.status_code == 429:
                st.warning("⚠️ Too many requests to Yahoo Finance. Please wait and try again.")
            elif search_response.status_code == 200:
                results = search_response.json().get("quotes", [])
                if results:
                    resolved_ticker = results[0].get("symbol", query)
                    st.session_state.ticker_cache[query] = resolved_ticker
                    st.success(f"✅ Found ticker: `{resolved_ticker}`")
                else:
                    st.warning("❌ Could not resolve a valid stock ticker.")
                    st.stop()
            else:
                st.warning(f"❌ Yahoo Finance search failed (Status {search_response.status_code}).")
                st.stop()

        # 📉 Price Chart
        if resolved_ticker is not None:
            st.markdown(f"#### 📉 Price Chart for {resolved_ticker}")
            stock = yf.Ticker(resolved_ticker)
            data = stock.history(period=period)
            if data.empty:
                st.error("No stock data found. Please check the ticker symbol.")
            else:
                st.line_chart(data["Close"])

            # 📰 News Headlines
            st.markdown(f"#### 📰 Recent News Headlines based on {resolved_ticker}")
            news_url = (
                f"https://newsapi.org/v2/everything?q={resolved_ticker}&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=publishedAt"
            )
            response = requests.get(news_url)
            headlines = []
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                if not articles:
                    st.info(f"ℹ️ No news articles found for `{resolved_ticker}`.")
                else:
                    for article in articles:
                        st.markdown(f"- [{article['title']}]({article['url']})")
                        headlines.append(article['title'])
            else:
                st.warning("❌ Failed to fetch relevant news articles.")

            # 🧠 Sentiment Analysis
            if headlines:
                st.markdown("#### 🧠 AI Sentiment Analysis")
                news_prompt = (
                    "Analyze the overall sentiment of these headlines. Return in 3 bullet points with a conclusion.\n\n"
                    + "\n".join(headlines)
                )
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = llm.invoke(news_prompt)
                st.success("🗣️ Sentiment Summary")
                st.markdown(sentiment_result)

    except Exception as e:
        st.error(f"Error during market analysis: {e}")

# === 📰 Trending Financial News (Configurable) ===
st.sidebar.markdown("---")
st.sidebar.markdown("### 📰 Trending News")

with st.sidebar.expander("📈 View Top Headlines"):
    # Country & Category selectors
    country_code = st.selectbox("🌍 Country", options={
        "India 🇮🇳": "in",
        "USA 🇺🇸": "us",
        "UK 🇬🇧": "gb",
        "Australia 🇦🇺": "au",
        "Canada 🇨🇦": "ca"
    }.keys(), index=0)
    selected_country = {
        "India 🇮🇳": "in",
        "USA 🇺🇸": "us",
        "UK 🇬🇧": "gb",
        "Australia 🇦🇺": "au",
        "Canada 🇨🇦": "ca"
    }[country_code]

    category = st.selectbox("📂 Category", ["business", "technology", "general", "science", "health", "sports", "entertainment"], index=0)

    try:
        trending_url = (
            f"https://newsapi.org/v2/top-headlines?"
            f"country={selected_country}&category={category}&apiKey={NEWS_API_KEY}&pageSize=5"
        )
        response = requests.get(trending_url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if articles:
                for idx, article in enumerate(articles, 1):
                    title = article["title"]
                    url = article["url"]
                    st.markdown(f"**{idx}.** [{title}]({url})")
            else:
                st.info("ℹ️ No trending news found for the selected filters.")
        else:
            st.warning(f"⚠️ Could not fetch news (Status {response.status_code})")
    except Exception as e:
        st.error(f"Error loading news: {e}")

# Cache embedding model
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()

# File Upload
uploaded_file = st.file_uploader("📎 Upload a financial document (PDF, CSV, XLSX)", type=["pdf", "csv", "xlsx"])

# Utility: Create hash for caching
def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Document type classifier
def infer_doc_type(text: str) -> str:
    lowered = text.lower()
    if any(kw in lowered for kw in ["account summary", "available balance", "transaction", "statement period"]):
        return "bank_statement"
    elif any(kw in lowered for kw in ["policyholder", "premium", "coverage", "sum assured"]):
        return "insurance_policy"
    elif any(kw in lowered for kw in ["financial year", "income statement", "cash flow", "balance sheet"]):
        return "annual_report"
    elif any(kw in lowered for kw in ["index methodology", "weighting scheme", "rebalancing"]):
        return "index_methodology"
    elif any(kw in lowered for kw in ["mutual fund", "nav", "expense ratio", "risk level"]):
        return "mutual_fund_factsheet"
    return "general"

# Prompt generator
def generate_summary_prompt(text: str) -> str:
    doc_type = infer_doc_type(text)
    base_prompt = {
        "bank_statement": "Summarize the bank statement. Highlight total credits and debits, average daily balance, high-value transactions, recurring payments, and suspicious activity if any.",
        "insurance_policy": "Summarize the insurance policy. Mention key details like policyholder name, sum assured, coverage duration, premium frequency, and exclusions.",
        "annual_report": "Summarize the company’s annual report. Focus on revenue, profit/loss trends, major expenditures, assets/liabilities, and future outlook.",
        "index_methodology": "Summarize the index methodology. Cover the calculation approach, weighting scheme, rebalancing frequency, inclusion/exclusion criteria, and special cases.",
        "mutual_fund_factsheet": "Summarize the mutual fund factsheet. Cover NAV trends, top holdings, fund objective, risk level, returns over time, and fund manager details.",
        "general": "Summarize this financial document in 5–7 bullet points. Include key figures, trends, clauses, metrics, and notable highlights."
    }
    return base_prompt.get(doc_type, base_prompt["general"]) + "\n\n" + text

# Prepare cache directory
cache_dir = "vector_cache"
os.makedirs(cache_dir, exist_ok=True)

if uploaded_file:
    file_suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if file_suffix in [".csv", ".xlsx"]:
        if file_suffix == ".csv":
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)

        st.subheader("📊 Uploaded Financial Data")
        st.dataframe(df)

        text_data = df.to_string(index=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.create_documents([text_data])

        file_hash = get_file_hash(tmp_path)
        db_path = os.path.join(cache_dir, file_hash)

        if os.path.exists(db_path):
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        else:
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(db_path)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type="stuff"
        )

        tab1, tab2= st.tabs(["🔍 Ask a Question", "🧠 Get a Summary"])

        with tab1:
            st.subheader("🔎 Ask AI about this data")
            query = st.text_input("Type your financial question", placeholder="e.g. What is the total revenue? Highlight quarterly trends.")

            sample_questions = [
                "What is the net profit trend?",
                "Summarize revenue and expense figures.",
                "Highlight major changes over time.",
                "What is the average revenue per quarter?",
                "Explain financial performance based on available data."
            ]

            col_buttons = st.columns(len(sample_questions))
            for idx, (col, question) in enumerate(zip(col_buttons, sample_questions)):
                if col.button(question, key=f"sample_csv_{idx}"):
                    query = question

            if query:
                with st.spinner("Generating answer..."):
                    answer = qa.run(query)
                st.success("Answer")
                st.write(answer)

        with tab2:
            st.subheader("📋 AI-Generated Data Summary")
            if st.button("Summarize Data"):
                full_text = text_data[:3500]
                summary_prompt = (
                    "Please analyze and summarize the financial spreadsheet below in 5-7 bullet points. "
                    "Highlight key metrics, trends, totals, averages, and any anomalies.\n\n" + full_text
                )
                with st.spinner("Summarizing data..."):
                    summary = llm.invoke(summary_prompt)
                st.success("📌 Summary")
                st.markdown(summary)

    elif file_suffix == ".pdf":
        file_hash = get_file_hash(tmp_path)
        db_path = os.path.join(cache_dir, file_hash)

        if os.path.exists(db_path):
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(pages)
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(db_path)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
            chain_type="stuff"
        )

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
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_documents(pages)
                full_text = "\n".join([c.page_content for c in chunks])[:3500]
                summary_prompt = generate_summary_prompt(full_text)
                with st.spinner("Summarizing document..."):
                    summary = llm.invoke(summary_prompt)
                st.success("📌 Summary")
                st.markdown(summary)

            with st.expander("📄 Preview Uploaded PDF"):
                with open(tmp_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

        try:
            os.remove(tmp_path)
        except:
            pass

        st.markdown("---")
        st.markdown("**ℹ️ Tip:** You can upload various types of financial documents — earnings reports, stock summaries, IPO filings, etc.")

else:
    st.markdown("<div class='file-upload-prompt'>👆 Please upload a financial document to begin analysis.</div>", unsafe_allow_html=True)
