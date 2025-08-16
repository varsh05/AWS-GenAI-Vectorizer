import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

# Load PDFs
loader = PyPDFDirectoryLoader(DATA_DIR)
documents = loader.load()

if not documents:
    raise RuntimeError(f"No PDFs found in ./{DATA_DIR}")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Create embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

# Create FAISS index
faiss_index = FAISS.from_documents(chunks, bedrock_embeddings)
faiss_index.save_local(INDEX_DIR)

print(f"✅ FAISS index created and saved in '{INDEX_DIR}'")
