# Sheria Intelligence Platform

> **Sheria** (Swahili) — *Law*

An AI-powered Kenyan legal intelligence platform that makes the law accessible to every citizen and professional — and holds the government accountable for public funds.

Built end-to-end as a production-grade data and AI engineering system: real-time document ingestion, multi-agent RAG, vector search, orchestrated pipelines, and a live public finance audit dashboard.

---

## What It Does

**For Citizens (Public Tier)**
Ask a legal question in plain English and get a clear, cited answer grounded in actual Kenyan law.

```
Query: "Can my employer fire me without notice?"

Response: Under the Employment Act 2007, your employer is generally 
required to give you notice before terminating your contract. For 
probationary contracts, the minimum notice period is seven days...

Sources: Employment Act 2007 — new.kenyalaw.org/akn/ke/act/2007/11
```

**For Professionals (Professional Tier)**
Submit a legal research query and receive a formatted legal brief with statutory references, cross-citations, and a Law Society of Kenya disclaimer.

**For Citizens Tracking Public Money**
A live Grafana dashboard showing every flagged audit finding, procurement irregularity, and budget absorption failure under the current government — by ministry, by county, in USD.

---

## Architecture

```
Kenya Law (new.kenyalaw.org)
          ↓
    scraper.py          Python + BeautifulSoup
          ↓
  kafka_producer.py     Apache Kafka (Docker)
          ↓
 spark_processor.py     Clean → Chunk (200 words, 30 overlap)
          ↓
   embeddings.py        Gemini gemini-embedding-001 (768 dims)
          ↓
  Aiven PostgreSQL       pgvector — vector similarity index
          ↓
   LangGraph Pipeline    4-agent chain
   ├── Research Agent   Domain detection + pgvector retrieval
   ├── Analysis Agent   Groq llama-3.3-70b legal reasoning
   ├── Citation Agent   Claim verification + retry routing
   └── Drafting Agent   Public chat or professional brief
          ↓
      FastAPI            REST API — /query /health /stats /domains
          ↓
      Grafana            Pipeline monitoring + Finance Audit Dashboard
          ↓
      Airflow            Daily 06:00 EAT ingestion orchestration
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | Python, BeautifulSoup, lxml |
| Message Queue | Apache Kafka |
| Processing | Custom Spark processor |
| Orchestration | Apache Airflow |
| Database | Aiven PostgreSQL 17 + pgvector 0.8.1 |
| Embeddings | Google Gemini `gemini-embedding-001` |
| LLM | Groq `llama-3.3-70b-versatile` |
| Agent Framework | LangGraph |
| API | FastAPI + Uvicorn |
| Monitoring | Grafana |
| Infrastructure | Docker Compose |

---

## Legal Knowledge Base

Six Acts across three priority domains — scraped directly from `new.kenyalaw.org` with versioned URLs that track amendments automatically.

| Domain | Act | Version |
|---|---|---|
| Employment | Employment Act 2007 | 2024-04-26 |
| Employment | Labour Relations Act 2007 | 2022-12-31 |
| Land | Land Act 2012 | 2022-12-31 |
| Land | Land Registration Act 2012 | 2022-12-31 |
| Business | Companies Act 2015 | 2022-12-31 |
| Business | Business Registration Service Act 2015 | 2022-12-31 |

**1,937 chunks** — 200 words each, 30-word overlap, indexed in pgvector for semantic similarity search.

---

## The Agent Pipeline

```
User Query
    ↓
Research Agent     Detects legal domain (employment/land/business)
                   Embeds query with Gemini retrieval_query task
                   Retrieves top 8 chunks via cosine similarity
    ↓
Analysis Agent     Reasons over retrieved provisions with Groq
                   Applies tier-appropriate depth (public vs professional)
                   Structures response: Answer → Legal Basis → Analysis → Implications
    ↓
Citation Agent     Verifies every legal claim traces to a retrieved source
                   Sets needs_retry=True if confidence is LOW
                   Builds citations list with source URLs
    ↓
[Retry loop]       If Citation Agent fails — broadens search scope
                   Max 2 retries before proceeding to drafting
    ↓
Drafting Agent     Public tier: plain-language conversational response
                   Professional tier: formatted legal brief with disclaimer
```

---

## Kenya Finance Audit Dashboard

A live Grafana dashboard tracking public finance irregularities under the current government (2022/23 — 2023/24).

**Current data:**
- **22 audit findings** across national ministries and all 47 counties
- **KES 92.56 billion (~$717M USD) flagged** by the Auditor General
- **5 procurement irregularity cases** with named suppliers
- **79 budget allocation records** covering absorption rates per entity

**Dashboard panels:**
- Total flagged funds (stat — hits you immediately)
- Top 10 national ministries with flagged funds
- Top 10 counties with flagged funds
- Flagged funds trend 2022/23 → 2023/24
- Types of financial irregularities (pie)
- Budget absorption rate by ministry
- Audit finding resolution status
- Cost per Kenyan citizen

**Data sources:** Office of the Auditor General, Office of the Controller of Budget, National Treasury, Appropriation Acts.

---

## API Reference

```
POST /query
{
  "query": "Can my employer terminate my contract without notice?",
  "user_tier": "public"    // or "professional"
}

GET /health     — Database + system health check
GET /stats      — Knowledge base statistics
GET /domains    — Available legal domains
```

**Response includes:** answer, citations with source URLs, domain detected, chunks retrieved, response time, retry count.

---

## Project Structure

```
sheria-intelligence/
├── ingestion/
│   ├── scraper.py              Kenya Law scraper
│   ├── kafka_producer.py       Kafka publisher with large doc splitting
│   └── spark_processor.py      Text cleaning, chunking, DB writer
├── store/
│   └── embeddings.py           Gemini embedding pipeline (idempotent)
├── agents/
│   ├── research_agent.py       Domain detection + pgvector retrieval
│   ├── analysis_agent.py       Groq legal analysis
│   ├── citation_agent.py       Claim verification + retry logic
│   ├── drafting_agent.py       Response formatting by tier
│   └── pipeline.py             LangGraph graph assembly
├── api/
│   └── main.py                 FastAPI application
├── pipeline/
│   └── dags/
│       └── sheria_dag.py       Airflow DAG
├── airflow/                    Airflow Docker setup
├── store/
├── eval/
├── monitoring/
├── docker-compose.yml          Kafka + Grafana infrastructure
├── requirements.txt
└── .env.example
```

---

## Setup & Running

### Prerequisites
- Docker + Docker Compose
- Python 3.12+
- Aiven PostgreSQL account (or any PostgreSQL 16+ with pgvector)
- Gemini API key (free tier — aistudio.google.com)
- Groq API key (free — console.groq.com)

### 1. Clone and configure

```bash
git clone https://github.com/KellyKiprop/sheria-intelligence.git
cd sheria-intelligence
cp .env.example .env
# Fill in your API keys and database credentials
```

### 2. Start infrastructure

```bash
# Start Kafka, Grafana
docker compose up -d

# Start Airflow
cd airflow && docker compose up -d
```

### 3. Set up virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Initialize database

```bash
# Enable pgvector and create schema
python3 store/embeddings.py --init-only
```

### 5. Run the ingestion pipeline

```bash
# Scrape + publish to Kafka
python3 ingestion/kafka_producer.py

# Process and chunk
python3 ingestion/spark_processor.py

# Generate embeddings (runs within Gemini free tier daily limit)
python3 store/embeddings.py
```

### 6. Start the API

```bash
python3 -m uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### 7. Automated pipeline (Airflow)

The Airflow DAG `sheria_legal_ingestion_pipeline` runs daily at 06:00 EAT automatically — scraping new Acts, publishing to Kafka, processing chunks, and generating embeddings for any new documents.

Access Airflow UI at `http://localhost:8080` (user: airflow / pass: airflow).

---

## Environment Variables

```env
# LLM & Embeddings
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# Database
DATABASE_URL=postgresql://user:password@host:port/db?sslmode=require
DB_HOST=your_host
DB_PORT=your_port
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_db

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_RAW_DOCS=sheria.raw.documents
KAFKA_TOPIC_PROCESSED=sheria.processed.chunks
```

---

## Key Engineering Decisions

**Why pgvector over a dedicated vector database?**
The legal knowledge base fits comfortably in PostgreSQL. pgvector with an IVFFlat index provides sub-100ms retrieval at our scale. Avoiding a separate vector DB reduces infrastructure complexity and keeps the stack familiar for the Kenyan teams likely to adopt this.

**Why LangGraph over CrewAI?**
Legal AI requires conditional routing — the Citation Agent needs to halt the pipeline and retry with broader retrieval if it detects unsupported claims. LangGraph's graph-based architecture supports this natively. CrewAI's sequential crew pattern doesn't.

**Why Groq for the LLM?**
Groq provides genuinely free inference on `llama-3.3-70b-versatile` with no daily token limits. For a public interest platform targeting Kenyan citizens, cost-free inference is essential for sustainability.

**Why 200-word chunks with 30-word overlap?**
A single legal provision — a section with its subsections — typically runs 100-250 words. At 200 words per chunk we capture one complete legal idea per chunk. The 30-word overlap ensures clauses at chunk boundaries appear fully in at least one chunk, which is critical for legal accuracy.

**Why versioned Kenya Law URLs?**
`new.kenyalaw.org` encodes the amendment date in every Act URL (`/eng@2024-04-26`). The Airflow DAG checks for newer amendment dates daily and ingests updated versions automatically — keeping the knowledge base current without manual intervention.

---

## Data Sources & Disclaimer

**Legal documents:** Kenya Law (`new.kenyalaw.org`) — official repository of Kenyan legislation. All documents are public domain.

**Financial data:** Office of the Auditor General Kenya, Office of the Controller of Budget, National Treasury. All figures sourced from official published reports.

**Disclaimer:** This platform provides legal information, not legal advice. For advice specific to your circumstances, consult a qualified advocate registered with the Law Society of Kenya. Financial audit data is sourced from official government reports and reflects findings by the Auditor General — not conclusions of guilt or criminal liability.

---

## Built By

**Kelly Kiprop** — Data Engineer  
Nairobi, Kenya  
[github.com/KellyKiprop](https://github.com/KellyKiprop)

*Built to make the law and public finance accountability accessible to every Kenyan.*
