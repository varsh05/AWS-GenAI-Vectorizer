# 🔍 AWS-GenAI-Vectorizer

A production-grade Generative AI application that integrates **AWS Bedrock**, **Amazon SageMaker**, **Hugging Face**, and **LangChain** to perform **text vectorization** and **question answering** using state-of-the-art foundation models.  
Deployed via a user-friendly **Streamlit interface**, the project is designed to showcase real-world applications of **LLMs** in enterprise search and NLP pipelines.

---

## 🚀 Why This Project Matters

Modern search engines, chatbots, and virtual assistants rely heavily on **text embeddings** and **contextual understanding**.  
This project replicates a real-world use case by combining:

- **Vectorization (via Titan Embeddings)** — for downstream retrieval, classification, and clustering tasks.
- **Question Answering (via SageMaker Deployed LLM)** — for extracting exact answers from custom user-provided contexts.

> ✅ Perfect for showcasing end-to-end GenAI and AWS skills in a single, clean, deployable package.

---

## 🧠 Core Highlights

| Module                     | Tech Stack                                      | Purpose                                           |
|----------------------------|--------------------------------------------------|---------------------------------------------------|
| `Text Embedding`           | AWS Bedrock (`amazon.titan-embed-text-v1`)     | Converts raw text into high-dimensional vectors   |
| `QA Model Deployment`      | Amazon SageMaker + Hugging Face Transformers   | Answers user queries from provided context        |
| `Orchestration`            | LangChain                                      | Prompts, pipelines, and embedding integration     |
| `Frontend`                 | Streamlit                                      | Simple interactive UI                             |

---

## 💻 Demo: What It Does

- 🔹 **Text Vectorizer Tab**  
  → Enter any sentence or paragraph and instantly generate its
