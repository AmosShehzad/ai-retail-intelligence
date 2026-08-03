# AI Retail Intelligence Assistant

An AI-powered business intelligence platform built for Pakistani kiryana (corner store) retailers, combining a RAG-powered chat assistant, sales analytics, and inventory intelligence into a single dashboard.

Ask questions in plain English such as "What should I restock this week?" or "Which category makes the most money?" and get answers grounded in actual store data rather than generic advice.

Live Demo: https://ai-retail-intelligence-4tuj.onrender.com
(Free tier deployment — first load may take 30-60 seconds to wake up.)

---

## Motivation

Most small retailers in Pakistan run their business on intuition, with no visibility into which products are dead stock, which categories drive profit, or when they are about to run out of a fast-selling item. This project turns raw transaction data into decisions a shopkeeper can act on, using the same architectural patterns found in production AI systems: retrieval-augmented generation, vector search, and provider-agnostic LLM integration.

---

## Features

**Store Overview**
Revenue, profit, margin, and product velocity, with interactive charts for daily, weekly, and monthly trends and category-level breakdowns.

**Inventory Intelligence**
Automatic dead-stock detection, low-stock alerts ranked by urgency (critical, high, medium), and restock recommendations with cost estimates.

**AI Assistant**
A RAG-powered chat interface. Every answer is grounded in retrieved store documents rather than model memory, with source citations and a live grounding score displayed for transparency.

**Manage Store**
Full CRUD for products and purchase orders. Placing an order and marking it received automatically updates stock levels, forming a complete procurement workflow rather than a demo form.

---

## Architecture

Streamlit Dashboard
|
FastAPI Backend (28 endpoints)
|
Analytics / Inventory Engines (Pandas)          RAG Pipeline
|                                                |
SQLite Database                                 Calibrated Retriever (FAISS)
|                                                |
query_logs                                       LLM (Groq or Ollama, swappable)

RAG pipeline flow: question is classified into a route (analytics, inventory, or base), calibrated FAISS retrieval runs with a top_k and similarity threshold specific to that route, retrieved documents are injected into the prompt as context, the LLM generates a response, and a hallucination/grounding check scores how well the answer traces back to the retrieved context before it is returned.

LLM provider abstraction: the RAG system behaves identically regardless of backend. Local development uses Ollama running phi3, fully offline. The deployed version uses Groq (Llama 3.1) for fast, free cloud inference. Switching between them requires changing a single environment variable, with no changes to the pipeline, retriever, or prompt logic.

---

## Tech Stack

Backend API: FastAPI, Pydantic, Uvicorn
Database: SQLite (products, sales, purchase_orders, query_logs)
RAG / AI: LangChain, FAISS, HuggingFace Sentence Transformers
LLM: Groq (Llama 3.1 8B) or Ollama (phi3), provider-agnostic
Frontend: Streamlit, Plotly
Observability: LangSmith (tracing and evaluation)
Containerization: Docker, Docker Compose
Deployment: Render

---

## Data

Synthetic but realistic Pakistani retail data covering 61 products across 10 categories, including Tea and Beverages, Spices and Masala, Dairy, Cooking Oil, Detergents, Personal Care, Instant Food, Snacks, Beverages, and Condiments. Prices are benchmarked against real 2025 Carrefour and Imtiaz Pakistan market rates. Sales data spans a full year with seasonal patterns built in, including Ramadan spikes for masala and Rooh Afza, summer spikes for cold drinks, and weekend and salary-week shopping boosts.

---

## Running Locally

Requirements: Python 3.11 or later, and either an Ollama installation for local inference or a free Groq API key for cloud inference.

git clone https://github.com/AmosShehzad/ai-retail-intelligence.git
cd ai-retail-intelligence

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create a .env file based on .env.example:

LLM_PROVIDER=ollama
GROQ_API_KEY=your_key_here
OLLAMA_MODEL=phi3
DB_PATH=retail.db

Generate the database and vector index:

python database/db_manager.py
python data/generate_realistic_data.py
python rag/embedder.py

Run the application:

python -m uvicorn api.app:app --reload --port 8000
streamlit run frontend/dashboard.py

Running with Docker (full stack, including local Ollama):

docker compose up -d
docker exec -it retail_ollama ollama pull phi3

---

## Key Engineering Decisions

Historical price accuracy: the sales table stores the price at time of transaction rather than joining against the current product price, so revenue history does not change retroactively when prices are updated.

Soft deletes: products are deactivated rather than hard-deleted, preserving referential integrity with historical sales records.

Calibrated retrieval: the retriever uses a different top_k and similarity threshold depending on question type. Inventory questions retrieve more documents at a looser threshold to catch every alert, while precise product lookups use a stricter threshold to avoid noise. This reduced irrelevant retrieval compared to a single fixed configuration.

Hallucination guarding: every answer is checked for keyword overlap with its retrieved context before being returned, with a grounding score surfaced in the interface so answer reliability is visible rather than assumed.

Provider abstraction for the LLM layer: moving from a self-hosted model to a hosted inference API required changing one environment variable and no pipeline code, because the LLM client is fully abstracted behind a single service class.

---

## Possible Improvements

Predictive stockout forecasting based on historical sales velocity trends
Fine-tuned product categorization instead of keyword-based rules
Multi-store support (currently single-tenant)

---

## License

MIT
