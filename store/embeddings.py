import os
import time
import logging
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # must match what we defined in chunk_embeddings table
BATCH_SIZE = 20             # how many chunks to embed per API call batch
RATE_LIMIT_DELAY = 1.0      # seconds between batches — respects Gemini free tier limits


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )


def fetch_unembedded_chunks(conn) -> list[tuple]:
    """
    Fetches all chunks that don't yet have an embedding.

    We use a LEFT JOIN to find chunks with no corresponding
    row in chunk_embeddings. This makes the function idempotent —
    safe to run multiple times without duplicating embeddings.
    Idempotent means: running it twice gives the same result as running it once.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT c.id, c.content, c.domain
        FROM document_chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE e.id IS NULL
        ORDER BY c.id;
    """)
    chunks = cur.fetchall()
    cur.close()
    return chunks


def embed_text(text: str) -> list[float]:
    """
    Sends a single text to Gemini and returns a 768-dim vector.

    task_type='retrieval_document' tells Gemini this text is a
    document being indexed — as opposed to a query being searched.
    This distinction matters: Gemini optimizes embeddings differently
    for documents vs queries, improving retrieval accuracy.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIMENSIONS
    )
    return result["embedding"]


def save_embedding(conn, chunk_id: int, embedding: list[float]):
    """Stores a single embedding vector linked to its chunk."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chunk_embeddings (chunk_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;
    """, (chunk_id, embedding))
    conn.commit()
    cur.close()


def embed_all_chunks() -> dict:
    """
    Main embedding function.

    Fetches all unembedded chunks, embeds them in batches,
    and stores each vector in chunk_embeddings.

    We process in batches and add a delay between batches
    to stay within Gemini's free tier rate limits (1500 req/day).
    """
    conn = get_db_connection()
    chunks = fetch_unembedded_chunks(conn)

    if not chunks:
        logger.info("All chunks already embedded. Nothing to do.")
        conn.close()
        return {"embedded": 0, "failed": 0}

    total = len(chunks)
    logger.info(f"Found {total} chunks to embed")
    logger.info(f"Estimated API calls: {total}")
    logger.info(f"Processing in batches of {BATCH_SIZE}\n")

    results = {"embedded": 0, "failed": 0, "failed_ids": []}

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Batch {batch_num}/{total_batches} — chunks {i+1} to {min(i+BATCH_SIZE, total)}")

        for chunk_id, content, domain in batch:
            try:
                embedding = embed_text(content)
                save_embedding(conn, chunk_id, embedding)
                results["embedded"] += 1

            except Exception as e:
                logger.error(f"  ❌ Failed chunk {chunk_id}: {e}")
                results["failed"] += 1
                results["failed_ids"].append(chunk_id)

        # Progress update every batch
        pct = (results["embedded"] / total) * 100
        logger.info(f"  Progress: {results['embedded']}/{total} ({pct:.1f}%)")

        # Rate limit delay between batches
        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)

    conn.close()
    return results


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Sheria Intelligence — Embeddings Pipeline")
    logger.info("=" * 60)

    results = embed_all_chunks()

    logger.info("\n" + "=" * 60)
    logger.info("📊 Embedding Summary")
    logger.info("=" * 60)
    logger.info(f"   Chunks embedded: {results['embedded']}")
    logger.info(f"   Failed:          {results['failed']}")
    if results.get("failed_ids"):
        logger.info(f"   Failed IDs: {results['failed_ids'][:10]}")
    logger.info("=" * 60)
