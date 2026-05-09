import os
import time
import logging
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import sys

# Add agents directory to path so we can import the pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from pipeline import run_pipeline

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Sheria Intelligence API",
    description="AI-powered Kenyan legal intelligence platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allows the API to be called from a browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The legal question to answer",
        example="Can my employer terminate my contract without notice?"
    )
    user_tier: str = Field(
        default="public",
        description="User tier: 'public' for plain language, 'professional' for legal brief",
        example="public"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Can my employer terminate my contract without notice?",
                "user_tier": "public"
            }
        }


class CitationResponse(BaseModel):
    title: str
    domain: str
    source_url: str
    similarity: float


class QueryResponse(BaseModel):
    query: str
    response: str
    citations: list[CitationResponse]
    domain: Optional[str]
    chunks_retrieved: int
    retries: int
    response_time_ms: int
    user_tier: str


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    domains: dict
    total_queries: int


# ── Database helper ───────────────────────────────────────────

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )


def log_query(
    query: str,
    user_tier: str,
    response_time_ms: int,
    chunks_retrieved: int
):
    """
    Logs every query to the query_logs table.
    This feeds the Grafana dashboard with real usage data.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_logs
            (query, user_tier, response_time_ms, chunks_retrieved)
            VALUES (%s, %s, %s, %s);
        """, (query, user_tier, response_time_ms, chunks_retrieved))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


# ── Routes ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {
        "name": "Sheria Intelligence API",
        "version": "1.0.0",
        "description": "AI-powered Kenyan legal intelligence",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Verifies the API is running and can reach the database.
    Used by monitoring systems and load balancers.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """
    Returns knowledge base statistics.
    Shows how many documents, chunks, and embeddings are indexed.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM legal_documents;")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
        total_embeddings = cur.fetchone()[0]

        cur.execute("""
            SELECT domain, COUNT(*) as chunk_count
            FROM document_chunks
            GROUP BY domain;
        """)
        domains = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM query_logs;")
        total_queries = cur.fetchone()[0]

        cur.close()
        conn.close()

        return StatsResponse(
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_embeddings=total_embeddings,
            domains=domains,
            total_queries=total_queries
        )

    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")


@app.post("/query", response_model=QueryResponse)
def query_legal(request: QueryRequest):
    """
    Main query endpoint — the heart of the API.

    Accepts a legal question and user tier, runs the full
    4-agent LangGraph pipeline, and returns a structured response
    with citations and metadata.

    Public tier: plain language answer
    Professional tier: formatted legal brief
    """
    logger.info(f"Query received: '{request.query}' [{request.user_tier}]")

    # Validate user tier
    if request.user_tier not in ["public", "professional"]:
        raise HTTPException(
            status_code=400,
            detail="user_tier must be 'public' or 'professional'"
        )

    start_time = time.time()

    try:
        # Run the full pipeline
        result = run_pipeline(request.query, request.user_tier)
        response_time_ms = int((time.time() - start_time) * 1000)

        # Format citations for response
        citations = [
            CitationResponse(
                title=c["title"],
                domain=c["domain"],
                source_url=c["source_url"],
                similarity=c["similarity"]
            )
            for c in (result.get("citations") or [])
        ]

        # Log to database for monitoring
        log_query(
            query=request.query,
            user_tier=request.user_tier,
            response_time_ms=response_time_ms,
            chunks_retrieved=len(result.get("chunks", []))
        )

        logger.info(f"Query completed in {response_time_ms}ms")

        return QueryResponse(
            query=request.query,
            response=result["response"],
            citations=citations,
            domain=result.get("domain"),
            chunks_retrieved=len(result.get("chunks", [])),
            retries=result.get("retry_count", 0),
            response_time_ms=response_time_ms,
            user_tier=request.user_tier
        )

    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.get("/domains")
def get_domains():
    """Returns the legal domains currently indexed in the knowledge base."""
    return {
        "domains": [
            {
                "id": "employment",
                "name": "Employment & Labour Law",
                "description": "Worker rights, termination, contracts, unions",
                "acts": ["Employment Act 2007", "Labour Relations Act 2007"]
            },
            {
                "id": "land",
                "name": "Land & Property Law",
                "description": "Land rights, registration, compulsory acquisition",
                "acts": ["Land Act 2012", "Land Registration Act 2012"]
            },
            {
                "id": "business",
                "name": "Business & Company Law",
                "description": "Company registration, directors, shareholders",
                "acts": ["Companies Act 2015", "Business Registration Service Act 2015"]
            }
        ]
    }

