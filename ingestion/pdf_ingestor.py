import os
import re
import time
import logging
import psycopg2
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Config ────────────────────────────────────────────────────
CHUNK_SIZE = 200        # words per chunk
CHUNK_OVERLAP = 30      # word overlap between chunks
BATCH_SIZE = 20         # embeddings per API call
RATE_LIMIT_DELAY = 1.0  # seconds between batches

# ── PDF Acts to ingest ────────────────────────────────────────
PDF_ACTS = [
    {
        "title": "Computer Misuse and Cybercrimes Act 2018",
        "path": "/home/kelly/Documents/sheria-intelligence/data/acts/cybercrime_act_2018.pdf",
        "domain": "cybercrime",
        "doc_type": "legislation",
        "source_url": "https://kenyalaw.org/akn/ke/act/2018/5",
        "version_date": "2018-05-16"
    },
    {
        "title": "Computer Misuse and Cybercrimes Amendment Act 2025",
        "path": "/home/kelly/Documents/sheria-intelligence/data/acts/cybercrime_act_2025.pdf",
        "domain": "cybercrime",
        "doc_type": "legislation",
        "source_url": "https://kenyalaw.org/akn/ke/act/2025/17",
        "version_date": "2025-10-21"
    }
]

# ── DB ────────────────────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )

# ── PDF extraction ────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    logger.info(f"Extracting text from: {path}")
    full_text = []
    with pdfplumber.open(path) as pdf:
        logger.info(f"  Pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)
    combined = "\n".join(full_text)
    logger.info(f"  Extracted {len(combined):,} characters")
    return combined

# ── Chunking ──────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Clean text
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    logger.info(f"  Created {len(chunks)} chunks")
    return chunks

# ── DB operations ─────────────────────────────────────────────
def document_exists(conn, title: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT id FROM legal_documents WHERE title = %s", (title,))
    exists = cur.fetchone() is not None
    cur.close()
    return exists

def insert_document(conn, act: dict) -> int:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO legal_documents (title, source_url, domain, doc_type)
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """, (act["title"], act["source_url"], act["domain"], act["doc_type"]))
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    logger.info(f"  Inserted document id={doc_id}")
    return doc_id

def insert_chunks(conn, doc_id: int, chunks: list[str], domain: str, doc_type: str) -> list[int]:
    cur = conn.cursor()
    chunk_ids = []
    for chunk in chunks:
        cur.execute("""
            INSERT INTO document_chunks (document_id, content, domain, doc_type)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (doc_id, chunk, domain, doc_type))
        chunk_ids.append(cur.fetchone()[0])
    conn.commit()
    cur.close()
    logger.info(f"  Inserted {len(chunk_ids)} chunks")
    return chunk_ids

def embed_and_store(conn, chunk_ids: list[int], chunks: list[str]):
    cur = conn.cursor()
    embedded = 0
    failed = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_ids = chunk_ids[i:i + BATCH_SIZE]
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        for chunk_id, chunk in zip(batch_ids, batch_chunks):
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=768
                    )
                )
                embedding = result.embeddings[0].values
                cur.execute("""
                    INSERT INTO chunk_embeddings (chunk_id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (chunk_id, embedding))
                embedded += 1
            except Exception as e:
                logger.error(f"  Embedding failed for chunk {chunk_id}: {e}")
                failed += 1

        conn.commit()
        pct = (embedded / len(chunks)) * 100
        logger.info(f"  Batch {batch_num}/{total_batches} — {embedded}/{len(chunks)} ({pct:.1f}%)")

        if i + BATCH_SIZE < len(chunks):
            time.sleep(RATE_LIMIT_DELAY)

    cur.close()
    logger.info(f"  Embedded: {embedded} | Failed: {failed}")

# ── Main ──────────────────────────────────────────────────────
def ingest_pdfs():
    conn = get_db_connection()

    for act in PDF_ACTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {act['title']}")
        logger.info(f"{'='*60}")

        # Skip if already ingested
        if document_exists(conn, act["title"]):
            logger.info(f"Already ingested — skipping")
            continue

        # Extract text
        text = extract_text_from_pdf(act["path"])
        if not text or len(text) < 500:
            logger.error(f"Insufficient text extracted — skipping")
            continue

        # Chunk
        chunks = chunk_text(text)

        # Insert document record
        doc_id = insert_document(conn, act)

        # Insert chunks
        chunk_ids = insert_chunks(conn, doc_id, chunks, act["domain"], act["doc_type"])

        # Embed and store
        logger.info(f"Embedding {len(chunks)} chunks...")
        embed_and_store(conn, chunk_ids, chunks)

        logger.info(f"Done: {act['title']}")

    conn.close()
    logger.info("\nIngestion complete")

if __name__ == "__main__":
    ingest_pdfs()
# This will be ignored — we're updating PDF_ACTS directly
