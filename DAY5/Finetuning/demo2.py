#!/usr/bin/env python3
"""
 Financial RAG Question‑Answering 
====================================================================
This script shows:
1. Loading PDF financial reports (10-K / 10-Q / Annual Reports)
2. Splitting text into chunks
3. Creating embeddings (SentenceTransformers)
4. Storing & running retrieval with FAISS
5. Building a simple RAG pipeline using LLaMA 3 (via transformers)

Dependencies to install:
    pip install langchain faiss-cpu transformers sentence-transformers pypdf
"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------
# 1. Load Financial PDF File
# ------------------------------------------------------------

PDF_PATH = "financial_report.pdf"  # Change to your PDF

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError("Please place your financial_report.pdf in this folder.")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} PDF pages.")

# ------------------------------------------------------------
# 2. Chunking Text
# ------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)
print(f"Chunked into {len(chunks)} chunks.")

# ------------------------------------------------------------
# 3. Create Vector DB using FAISS
# ------------------------------------------------------------

print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_db = FAISS.from_documents(chunks, embeddings)
print("Vector DB created.")

# ------------------------------------------------------------
# 4. Load LLM (LLaMA-3 or any small model)
# ------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ------------------------------------------------------------
# 5. RAG Query Function
# ------------------------------------------------------------

def ask_rag(query, k=4):
    """Retrieve top-k chunks and run RAG inference."""

    retrieved = vector_db.similarity_search(query, k=k)
    context = "
".join([d.page_content for d in retrieved])

    prompt = f"""
You are a financial analysis assistant.
Use ONLY the context below to answer the question.
If context is not enough, say 'Not available in the document.'

Context:
{context}

Question: {query}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------------------------------------
# 6. Test Query
# ------------------------------------------------------------

if __name__ == "__main__":
    print("RAG System Ready! Ask something like:")
    print(" - What was the revenue growth?
 - What were the risk factors?
 - Summarize the management discussion.")

    while True:
        q = input("Ask a financial question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = ask_rag(q)
        print("--------------------Answer:", answer)
