import logging
import re
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from kafka import KafkaConsumer

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Chunk settings ---
# 200 words is the sweet spot for legal RAG:
# specific enough to represent one legal concept,
# large enough to contain a full clause or provision.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

# --- Noise patterns ---
# Everything here is UI chrome injected by Kenya Law's website.
# None of it is legal content — it corrupts embeddings if left in.
UI_NOISE_PATTERNS = [
    r"Download PDF.*?KB\)",
    r"Report\s+Report a problem",
    r"Report a problem",
    r"Copy citation",
    r"Document detail",
    r"Related documents",
    r"Subsidiary legislation",
    r"Citations?\s+\d+\s*/\s*\d+",
    r"\d+\s*/\s*\d+",
    r"This is the latest version of this Act\.",
    r"This is the version of this Act as it was from.*?\.",
    r"Read the latest available version\s*\.",
    r"Cap\.\s*\d+[A-Z]?",
    r"Copy\s+Date\s+\d{1,2}\s+\w+\s+\d{4}",
    r"Print this page",
    r"Share this",
    r"Back to top",
    r"(?i)next\s+page",
    r"(?i)previous\s+page",
    r"(?i)table of contents",
]


def clean_text(raw_text: str) -> str:
    """
    Strips UI noise from scraped legal text and normalizes whitespace.

    The goal is to leave only the actual legal content —
    section numbers, clause text, definitions, provisions.
    Anything the Kenya Law website renders as navigation or
    metadata needs to go before we embed.
    """
    text = raw_text

    for pattern in UI_NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    text = re.sub(r" {2,}", " ", text)         # max 1 consecutive space
    text = re.sub(r"\t+", " ", text)           # tabs to spaces
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits legal text into overlapping word-based chunks.

    Why 200 words?
    A single legal provision — a section with its subsections —
    typically runs 100-250 words. At 200 words per chunk we capture
    one complete legal idea per chunk, which is exactly what we want
    for precise semantic retrieval.

    Why 30-word overlap?
    Provisions that span chunk boundaries will appear fully in at
    least one chunk. Without overlap, a clause split across two chunks
    would be incomplete in both — making retrieval inaccurate.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        # Skip near-empty chunks — they add noise without value
        if len(chunk.strip()) > 30:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_db_connection():
    """Returns a psycopg2 connection to Aiven PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        sslmode="require"
    )


def save_document(conn, title: str, source_url: str, domain: str,
                  doc_type: str, raw_content: str) -> int | None:
    """
    Inserts a document into legal_documents.
    Returns the new document ID or None if it already exists.

    ON CONFLICT DO NOTHING means re-running the processor
    never creates duplicate documents — idempotent by design.
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO legal_documents (title, source_url, domain, doc_type, raw_content, is_processed)
        VALUES (%s, %s, %s, %s, %s, FALSE)
        ON CONFLICT DO NOTHING
        RETURNING id;
    """, (title, source_url, domain, doc_type, raw_content))

    result = cur.fetchone()
    conn.commit()
    cur.close()
    return result[0] if result else None


def save_chunks(conn, document_id: int, chunks: list[str],
                domain: str, doc_type: str, source_url: str) -> int:
    """
    Bulk inserts all chunks for a document into document_chunks.
    Returns the number of chunks saved.
    """
    cur = conn.cursor()
    saved = 0

    for i, chunk in enumerate(chunks):
        cur.execute("""
            INSERT INTO document_chunks
            (document_id, chunk_index, content, domain, doc_type, source_url)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (document_id, i, chunk, domain, doc_type, source_url))
        saved += 1

    conn.commit()
    cur.close()
    return saved


def mark_document_processed(conn, document_id: int):
    """Marks a document as fully processed."""
    cur = conn.cursor()
    cur.execute(
        "UPDATE legal_documents SET is_processed = TRUE WHERE id = %s;",
        (document_id,)
    )
    conn.commit()
    cur.close()


def reassemble_parts(messages: list[dict]) -> dict:
    """
    Groups Kafka messages by source URL and reassembles
    multi-part documents (like the Companies Act which was
    split into 2 messages due to Kafka's size limit).

    Returns a dict keyed by source_url with full content.
    """
    doc_parts = {}

    for msg in messages:
        url = msg["source_url"]
        if url not in doc_parts:
            doc_parts[url] = {
                "title": msg["title"],
                "source_url": msg["source_url"],
                "domain": msg["domain"],
                "doc_type": msg["doc_type"],
                "version_date": msg["version_date"],
                "parts": {}
            }
        doc_parts[url]["parts"][msg["part_index"]] = msg["raw_content"]

    # Sort parts by index and join into full content
    for url in doc_parts:
        sorted_parts = [
            doc_parts[url]["parts"][i]
            for i in sorted(doc_parts[url]["parts"].keys())
        ]
        doc_parts[url]["full_content"] = "\n".join(sorted_parts)

    return doc_parts


def process_messages(messages: list[dict]) -> dict:
    """
    Core processing function:
    1. Reassemble multi-part documents
    2. Clean raw text
    3. Chunk into 200-word overlapping segments
    4. Save document + chunks to PostgreSQL
    5. Mark document as processed
    """
    doc_parts = reassemble_parts(messages)

    results = {
        "documents_processed": 0,
        "total_chunks": 0,
        "failed": []
    }

    conn = get_db_connection()

    for url, doc_data in doc_parts.items():
        try:
            raw = doc_data["full_content"]
            title = doc_data["title"]

            logger.info(f"Processing: {title}")
            logger.info(f"  Raw:     {len(raw):,} chars")

            # Step 1: Clean
            cleaned = clean_text(raw)
            logger.info(f"  Cleaned: {len(cleaned):,} chars")

            # Step 2: Chunk
            chunks = chunk_text(cleaned)
            logger.info(f"  Chunks:  {len(chunks)}")

            # Step 3: Save document
            doc_id = save_document(
                conn,
                title=title,
                source_url=doc_data["source_url"],
                domain=doc_data["domain"],
                doc_type=doc_data["doc_type"],
                raw_content=cleaned
            )

            if not doc_id:
                logger.warning(f"  ⚠️  Already exists, skipping: {title}")
                continue

            # Step 4: Save chunks
            saved = save_chunks(
                conn,
                document_id=doc_id,
                chunks=chunks,
                domain=doc_data["domain"],
                doc_type=doc_data["doc_type"],
                source_url=doc_data["source_url"]
            )

            # Step 5: Mark processed
            mark_document_processed(conn, doc_id)

            logger.info(f"  ✅ Done — {saved} chunks saved\n")
            results["documents_processed"] += 1
            results["total_chunks"] += saved

        except Exception as e:
            logger.error(f"  ❌ Failed: {doc_data['title']} — {e}")
            results["failed"].append(doc_data["title"])

    conn.close()
    return results


if __name__ == "__main__":
    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    logger.info("=" * 60)
    logger.info("Sheria Intelligence — Spark Processor")
    logger.info(f"Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")
    logger.info(f"Consuming from: {topic}")
    logger.info("=" * 60 + "\n")

    # Consume all messages from Kafka topic
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="sheria-processor-v2",     # new group ID — reads from the beginning
        consumer_timeout_ms=5000
    )

    messages = []
    for msg in consumer:
        messages.append(msg.value)
    consumer.close()

    logger.info(f"Consumed {len(messages)} messages from Kafka\n")

    if not messages:
        logger.warning("No messages found. Run kafka_producer.py first.")
        exit(0)

    results = process_messages(messages)

    logger.info("=" * 60)
    logger.info("📊 Processing Summary")
    logger.info("=" * 60)
    logger.info(f"   Documents processed: {results['documents_processed']}")
    logger.info(f"   Total chunks saved:  {results['total_chunks']}")
    logger.info(f"   Failed:              {len(results['failed'])}")
    if results["failed"]:
        logger.info(f"   Failed: {results['failed']}")
    logger.info("=" * 60)
