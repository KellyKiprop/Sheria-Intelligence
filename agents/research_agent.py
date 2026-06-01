import os
import logging
import psycopg2
from google import genai
from google.genai import types
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Gemini Client ─────────────────────────────────────────────
# New SDK uses a client instance instead of module-level configure()
_client = None

def get_genai_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

# ── Shared State ─────────────────────────────────────────────
# This TypedDict is the baton passed between all agents.
# Every agent reads from it and writes its results back into it.
# TypedDict means Python knows exactly what keys and types to expect.
class SheriaState(TypedDict):
    query: str                          # original user question
    domain: Optional[str]              # detected legal domain
    user_tier: str                     # "public" or "professional"
    chunks: list[dict]                 # retrieved chunks (Research fills)
    analysis: Optional[str]           # legal reasoning (Analysis fills)
    citations: Optional[list[dict]]   # verified sources (Citation fills)
    response: Optional[str]           # final answer (Drafting fills)
    needs_retry: bool                  # Citation sets True if claims fail

# ── Retrieval ────────────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )

def embed_query(query: str) -> list[float]:
    """
    Embeds the user query using Gemini.
    Note: task_type='RETRIEVAL_QUERY' — different from 'RETRIEVAL_DOCUMENT'
    used when indexing. Gemini optimizes query embeddings differently
    from document embeddings to improve retrieval accuracy.
    This asymmetric embedding is why our retrieval will be more precise
    than naive approaches that embed queries and documents the same way.
    """
    result = get_genai_client().models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values

def detect_domain(query: str) -> Optional[str]:
    """
    Detects which legal domain a query belongs to.
    This is a simple keyword-based classifier.
    In a future version this could be a proper ML classifier
    or an LLM call — but keyword matching is fast, free,
    and accurate enough for our three domains.
    Returning None means the query spans multiple domains
    and we search across everything.
    """
    query_lower = query.lower()

    employment_keywords = [
        "employ", "fired", "terminate", "salary", "wage", "contract",
        "leave", "maternity", "redundancy", "dismiss", "resign", "notice",
        "worker", "employee", "employer", "labour", "union", "strike"
    ]
    land_keywords = [
        "land", "property", "title deed", "plot", "lease", "tenant",
        "landlord", "evict", "ownership", "survey", "cadastral", "acre",
        "hectare", "caution", "adverse possession", "compulsory acquisition"
    ]
    business_keywords = [
        "company", "business", "director", "shareholder", "register",
        "incorporation", "contract", "liability", "partnership", "shares",
        "dividend", "memorandum", "articles", "debenture", "liquidation"
    ]

    employment_score = sum(1 for k in employment_keywords if k in query_lower)
    land_score = sum(1 for k in land_keywords if k in query_lower)
    business_score = sum(1 for k in business_keywords if k in query_lower)

    scores = {
        "employment": employment_score,
        "land": land_score,
        "business": business_score
    }

    best_domain = max(scores, key=scores.get)

    # Only return a domain if there's a clear signal
    # If all scores are 0 or tied, search everything
    if scores[best_domain] == 0:
        return None

    return best_domain

def retrieve_chunks(
    query_embedding: list[float],
    domain: Optional[str],
    top_k: int = 8
) -> list[dict]:
    """
    Retrieves the most semantically similar chunks from pgvector.
    Why top_k=8?
    Too few (3-4) and we might miss relevant provisions.
    Too many (15+) and we flood the LLM with noise, hurting analysis quality.
    8 is the sweet spot for legal queries — enough coverage, tight enough focus.
    The cosine distance operator <=> finds vectors closest in direction
    regardless of magnitude — ideal for semantic similarity.
    If a domain is detected we filter to that domain first,
    then fall back to all domains if fewer than 3 results come back.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    if domain:
        cur.execute("""
            SELECT
                c.id,
                c.content,
                c.domain,
                c.doc_type,
                c.source_url,
                d.title,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.id = e.chunk_id
            JOIN legal_documents d ON d.id = c.document_id
            WHERE c.domain = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, domain, query_embedding, top_k))
    else:
        cur.execute("""
            SELECT
                c.id,
                c.content,
                c.domain,
                c.doc_type,
                c.source_url,
                d.title,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.id = e.chunk_id
            JOIN legal_documents d ON d.id = c.document_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))

    rows = cur.fetchall()

    # If domain filter returned too few results, go broader
    if domain and len(rows) < 3:
        logger.info(f"Domain filter returned {len(rows)} results — expanding to all domains")
        cur.execute("""
            SELECT
                c.id,
                c.content,
                c.domain,
                c.doc_type,
                c.source_url,
                d.title,
                1 - (e.embedding <=> %s::vector) AS similarity
            FROM chunk_embeddings e
            JOIN document_chunks c ON c.id = e.chunk_id
            JOIN legal_documents d ON d.id = c.document_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))
        rows = cur.fetchall()

    cur.close()
    conn.close()

    chunks = []
    for row in rows:
        chunks.append({
            "chunk_id": row[0],
            "content": row[1],
            "domain": row[2],
            "doc_type": row[3],
            "source_url": row[4],
            "title": row[5],
            "similarity": round(row[6], 4)
        })

    return chunks

# ── Research Agent Node ───────────────────────────────────────
def research_agent(state: SheriaState) -> SheriaState:
    """
    Research Agent — Node 1 in the LangGraph pipeline.
    Responsibilities:
    1. Detect which legal domain the query belongs to
    2. Embed the query using Gemini
    3. Retrieve the top 8 most relevant chunks from pgvector
    4. Log what was found and how confident we are
    5. Return updated state with chunks filled in
    This agent never calls the LLM — it's purely retrieval.
    Keeping retrieval and reasoning in separate agents means
    we can tune, test, and improve each independently.
    """
    query = state["query"]
    logger.info(f"Research Agent — Query: '{query}'")

    # Step 1: Detect domain
    domain = detect_domain(query)
    logger.info(f"Detected domain: {domain or 'all domains'}")

    # Step 2: Embed query
    query_embedding = embed_query(query)
    logger.info("Query embedded successfully")

    # Step 3: Retrieve
    chunks = retrieve_chunks(query_embedding, domain, top_k=8)
    logger.info(f"Retrieved {len(chunks)} chunks")

    # Step 4: Log quality signal
    if chunks:
        top_similarity = chunks[0]["similarity"]
        logger.info(f"Top similarity score: {top_similarity}")
        if top_similarity < 0.6:
            logger.warning(
                f"Low similarity ({top_similarity}) — query may be outside "
                f"current knowledge base"
            )
        for i, chunk in enumerate(chunks[:3], 1):
            logger.info(
                f"  Result {i}: {chunk['title']} "
                f"[{chunk['domain']}] sim={chunk['similarity']}"
            )

    # Step 5: Return updated state
    return {
        **state,
        "domain": domain,
        "chunks": chunks
    }

# ── Standalone Test ───────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        {
            "query": "Can my employer terminate my contract without notice?",
            "user_tier": "public"
        },
        {
            "query": "What are my rights if my landlord evicts me illegally?",
            "user_tier": "public"
        },
        {
            "query": "What are the requirements to register a company in Kenya?",
            "user_tier": "professional"
        }
    ]

    for test in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {test['query']}")
        print('='*65)

        initial_state: SheriaState = {
            "query": test["query"],
            "domain": None,
            "user_tier": test["user_tier"],
            "chunks": [],
            "analysis": None,
            "citations": None,
            "response": None,
            "needs_retry": False
        }

        result = research_agent(initial_state)

        print(f"\nDomain detected: {result['domain']}")
        print(f"Chunks retrieved: {len(result['chunks'])}")
        print(f"\nTop 3 results:")
        for i, chunk in enumerate(result["chunks"][:3], 1):
            print(f"\n  {i}. {chunk['title']} [{chunk['domain']}]")
            print(f"     Similarity: {chunk['similarity']}")
            print(f"     Preview: {chunk['content'][:200]}...")
