# app.py
import os
import streamlit as st
from vector_store import create_vector_store, load_vector_store, DATA_DIR, INDEX_DIR, ensure_dirs
from rag_pipeline import run_rag
from models.claude import get_claude_llm
from models.llama2 import get_llama2_llm

# Streamlit Page Config
st.set_page_config(page_title="Chat with PDF | Bedrock RAG", layout="wide")
st.title("💬 Chat with your PDFs — AWS Bedrock + LangChain (RAG)")

# Ensure required directories exist
ensure_dirs()

# Sidebar - Document Management
with st.sidebar:
    st.header("📄 Manage Documents")

    # Upload PDFs
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
        st.success(f"✅ Uploaded {len(uploaded)} file(s) to '{DATA_DIR}'")

    # Build / Refresh Vector Index
    if st.button("🔄 Build / Refresh Vector Index"):
        with st.spinner("🔍 Embedding and indexing documents..."):
            try:
                create_vector_store(DATA_DIR, INDEX_DIR)
                st.success("✅ Vector index updated successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Show available PDFs
    st.subheader("📂 Current PDFs")
    pdf_list = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if pdf_list:
        for pdf in pdf_list:
            st.write(f"📎 {pdf}")
    else:
        st.info("No PDFs found in 'data' folder.")

# Main Chat Interface
st.subheader("💬 Ask a question about your documents")
query = st.text_input("Your question:")

col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox("Choose Model", ["Claude (Anthropic)", "Llama2 (Meta)"])
with col2:
    top_k = st.slider("Retriever top-k", min_value=1, max_value=10, value=3)

if st.button("✨ Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        try:
            with st.spinner("📂 Loading vector store..."):
                # Rebuild index if missing or empty
                if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
                    st.info("📂 FAISS index not found. Creating new index from PDFs...")
                    create_vector_store(DATA_DIR, INDEX_DIR)

                vs = load_vector_store(INDEX_DIR)

            with st.spinner("🤖 Generating answer..."):
                llm = get_claude_llm() if model_choice.startswith("Claude") else get_llama2_llm()
                answer = run_rag(llm, vs, query, k=top_k)

            st.success("✅ Done")
            st.write(answer)

        except FileNotFoundError:
            st.error("⚠️ Vector index not found! Please upload PDFs first.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
