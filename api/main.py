import os
import hashlib
import json
import time
import logging
import psycopg2
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
from pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Sheria Intelligence API",
    description="AI-powered Kenyan legal intelligence platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────
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
        description="'public' for plain language, 'professional' for legal brief",
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


# ── DB helpers ────────────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )


def log_query(query: str, user_tier: str, response_time_ms: int, chunks_retrieved: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_logs (query, user_tier, response_time_ms, chunks_retrieved)
            VALUES (%s, %s, %s, %s);
        """, (query, user_tier, response_time_ms, chunks_retrieved))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


# ── Cache helpers ─────────────────────────────────────────────
def get_cache_key(query: str, user_tier: str) -> str:
    normalized = f"{query.strip().lower()}:{user_tier}"
    return hashlib.md5(normalized.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[dict]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT response, citations, domain, chunks_retrieved
            FROM query_cache
            WHERE cache_key = %s AND expires_at > NOW()
        """, (cache_key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "response": row[0],
                "citations": row[1],
                "domain": row[2],
                "chunks_retrieved": row[3]
            }
        return None
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None


def save_to_cache(cache_key: str, query: str, user_tier: str, result: dict):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_cache (cache_key, query, user_tier, response, citations, domain, chunks_retrieved)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE SET
                response = EXCLUDED.response,
                citations = EXCLUDED.citations,
                domain = EXCLUDED.domain,
                chunks_retrieved = EXCLUDED.chunks_retrieved,
                created_at = NOW(),
                expires_at = NOW() + INTERVAL '24 hours'
        """, (
            cache_key,
            query,
            user_tier,
            result.get("response", ""),
            json.dumps(result.get("citations", [])),
            result.get("domain"),
            len(result.get("chunks", []))
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Response cached with key: {cache_key[:8]}...")
    except Exception as e:
        logger.error(f"Cache save error: {e}")


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
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"disconnected: {str(e)}"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM legal_documents;")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
        total_embeddings = cur.fetchone()[0]

        cur.execute("SELECT domain, COUNT(*) FROM document_chunks GROUP BY domain;")
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


@app.get("/domains")
def get_domains():
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


@app.post("/query", response_model=QueryResponse)
def query_legal(request: QueryRequest):
    logger.info(f"Query received: '{request.query}' [{request.user_tier}]")

    if request.user_tier not in ["public", "professional"]:
        raise HTTPException(
            status_code=400,
            detail="user_tier must be 'public' or 'professional'"
        )

    # Check cache first
    cache_key = get_cache_key(request.query, request.user_tier)
    cached = get_cached_response(cache_key)

    if cached:
        logger.info(f"Cache HIT for key: {cache_key[:8]}...")
        citations = []
        if cached["citations"]:
            raw = cached["citations"] if isinstance(cached["citations"], list) else json.loads(cached["citations"])
            citations = [
                CitationResponse(
                    title=c["title"],
                    domain=c["domain"],
                    source_url=c["source_url"],
                    similarity=c["similarity"]
                )
                for c in raw
            ]
        return QueryResponse(
            query=request.query,
            response=cached["response"],
            citations=citations,
            domain=cached["domain"],
            chunks_retrieved=cached["chunks_retrieved"],
            retries=0,
            response_time_ms=0,
            user_tier=request.user_tier
        )

    logger.info("Cache MISS — running pipeline")
    start_time = time.time()

    try:
        result = run_pipeline(request.query, request.user_tier)
        response_time_ms = int((time.time() - start_time) * 1000)

        citations = [
            CitationResponse(
                title=c["title"],
                domain=c["domain"],
                source_url=c["source_url"],
                similarity=c["similarity"]
            )
            for c in (result.get("citations") or [])
        ]

        log_query(
            query=request.query,
            user_tier=request.user_tier,
            response_time_ms=response_time_ms,
            chunks_retrieved=len(result.get("chunks", []))
        )

        logger.info(f"Query completed in {response_time_ms}ms")

        # Save to cache
        save_to_cache(cache_key, request.query, request.user_tier, result)

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
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
