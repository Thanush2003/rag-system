# RAG-Based PDF Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based Question Answering system for PDF documents using semantic search and local Large Language Model inference.

The system:
- loads and chunks PDF documents
- generates semantic embeddings
- stores vectors in Milvus
- retrieves relevant chunks using cosine similarity
- generates grounded answers using Llama3 through Ollama
- streams responses to the frontend in real time

The project also includes:
- Recall@3 evaluation
- manual answer scoring
- metadata-aware retrieval
- out-of-corpus filtering

---

# Features

- PDF ingestion pipeline
- Semantic vector search
- Milvus vector database
- Metadata storage (source + page)
- Retrieval-Augmented Generation (RAG)
- Streaming responses
- Multi-document retrieval
- Recall@3 evaluation
- Out-of-corpus rejection
- FastAPI backend
- Lightweight frontend

---

# Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline where PDF documents are loaded and chunked during ingestion. Semantic embeddings are generated using SentenceTransformers and stored in Milvus along with metadata. During querying, the user question is embedded and relevant chunks are retrieved using cosine similarity search. The retrieved context is inserted into a prompt template and passed to the Llama3 model through Ollama for answer generation. Responses are streamed back to the frontend using FastAPI StreamingResponse.

Pipeline Flow:

PDFs → Chunking → Embeddings → Milvus → Retrieval → Prompting → Llama3 → Streaming Response

---

# Tech Stack

## Backend
- FastAPI
- Python
- Ollama
- Llama3

## Retrieval
- SentenceTransformers
- Milvus
- LangChain

## Frontend
- HTML
- CSS
- JavaScript

---


# Setup

## 1. Clone Repository

git clone <repo-url>

cd <project-folder>

---

## 2. Create Virtual Environment

Windows:

python -m venv venv

venv\Scripts\activate

---

## 3. Install Requirements

pip install -r requirements.txt

---

## 4. Install Docker Desktop

---

## 5. Install Ollama

Pull model:

ollama pull llama3

Run model:

ollama run llama3

---

## 6. Start Milvus

docker compose up -d

---

## 7. Run Ingestion

make ingest

---

## 8. Start Backend

make serve

---

## 9. Start Frontend

make frontend

---

## 10. Open Application

http://127.0.0.1:5500