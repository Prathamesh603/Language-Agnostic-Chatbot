import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv
from groq import Groq 

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
st.set_page_config(page_title="Multilingual PDF Chatbot", page_icon="💬", layout="wide")

st.sidebar.title("⚙️ Settings") 
default_model = "llama-3.1-8b-instant"
groq_model = st.sidebar.text_input("Groq model", value=default_model)
top_k = st.sidebar.slider("Retriever top-k", 2, 12, 5, 1)
chunk_size = st.sidebar.slider("Chunk size (chars)", 100, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", 50, 400, 150, 25)
temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

# Read API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.sidebar.warning("No GROQ_API_KEY found in environment/.env")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Global state
if "faiss" not in st.session_state:
    st.session_state.faiss = None 
if "docs" not in st.session_state:
    st.session_state.docs = []
if "emb" not in st.session_state:
    st.session_state.emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Helper functions
# ----------------------------
SYSTEM_PROMPT = """\
You are a helpful research assistant. Always answer using only the PDF context.
If context is insufficient, say so politely. Be concise, structured, and helpful.
"""

def build_context(snippets: List) -> str:
    blocks = []
    for i, d in enumerate(snippets, start=1):
        page = d.metadata.get("page", d.metadata.get("source", ""))
        label = d.metadata.get("page_label", page)
        blocks.append(f"[Source {i} | Page {label}]\n{d.page_content}")
    return "\n\n".join(blocks)

def translate_query_with_llm(user_query: str) -> str:
    """Use LLM to translate any language query into English."""
    if groq_client is None:
        return user_query  # fallback
    prompt = f"""
You are a translation assistant.
Translate the following question into clear, concise English for document retrieval.

Question: "{user_query}"
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that translates questions into English."},
        {"role": "user", "content": prompt}
    ]
    resp = groq_client.chat.completions.create(
        model=groq_model,
        messages=messages,
        temperature=0.0,
        stream=False
    )
    return resp.choices[0].message.content

def ask_groq(question: str, context: str) -> str:
    if groq_client is None:
        return "GROQ_API_KEY not configured."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}"}
    ]
    resp = groq_client.chat.completions.create(
        model=groq_model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    answer = ""
    placeholder = st.empty()
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
        placeholder.markdown(answer)
    return answer

def index_pdf(uploaded_file, chunk_size=1000, chunk_overlap=150):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    docs = splitter.split_documents(pages)
    for i, d in enumerate(docs):
        d.metadata["chunk_index"] = i
    index = FAISS.from_documents(docs, st.session_state.emb)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return index, docs

# ----------------------------
# UI
# ----------------------------
st.title("💬 Multilingual PDF Chatbot (Groq + FAISS)")
st.caption("Ask questions in English, Hindi, Gujarati, or Marathi. LLM will translate and retrieve context from the PDF.")

uploaded = st.file_uploader("📂 Upload a PDF", type=["pdf"])

colA, colB = st.columns(2)
with colA:
    if st.button("🔧 Build Index", type="primary", disabled=uploaded is None):
        if uploaded:
            with st.spinner("Indexing PDF…"):
                faiss_db, docs = index_pdf(uploaded, chunk_size, chunk_overlap)
                st.session_state.faiss = faiss_db
                st.session_state.docs = docs
            st.success(f"Indexed {len(docs)} chunks.")
with colB:
    if st.button("🧹 Clear Index"):
        st.session_state.faiss = None
        st.session_state.docs = []
        st.session_state.messages = []
        st.success("Index cleared.")

st.divider()

# ----------------------------
# Chat Interface
# ----------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

query = st.chat_input("Type your question...")

if query:
    # 1️⃣ Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    # 2️⃣ Translate query using LLM
    query_en = translate_query_with_llm(query)

    # 3️⃣ Retrieve context from FAISS
    if st.session_state.faiss:
        with st.spinner("Retrieving context…"):
            retrieved = st.session_state.faiss.similarity_search(query_en, k=top_k)
        context_text = build_context(retrieved)

        # 4️⃣ Get final answer from Groq
        answer = ask_groq(query_en, context_text)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload a PDF and build index first.")
