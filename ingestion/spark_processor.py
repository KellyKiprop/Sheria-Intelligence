import logging
import re
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Noise patterns to strip from scraped legal text ---
# These are UI artifacts that Kenya Law's website injects
# into the page alongside the actual legal content.
UI_NOISE_PATTERNS = [
    r"Download PDF.*?KB\)",
    r"Report\s+Report a problem",
    r"Copy citation",
    r"Document detail",
    r"History",
    r"Related documents",
    r"Subsidiary legislation",
    r"Citations?\s+\d+\s*/\s*\d+",
    r"This is the latest version of this Act\.",
    r"This is the version of this Act as it was from.*?\.",
    r"Read the latest available version\s*\.",
    r"Cap\.\s*\d+[A-Z]?",
    r"Copy\s+Date\s+\d{1,2}\s+\w+\s+\d{4}",
]


def clean_text(raw_text: str) -> str:
    """
    Removes UI noise from scraped legal text.

    We apply regex patterns to strip navigation elements,
    download buttons, and metadata injected by the website.
    Then we normalize whitespace so the text is clean
    and consistent before chunking.
    """
    text = raw_text

    for pattern in UI_NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits cleaned legal text into overlapping word-based chunks.

    Why word-based and not character-based?
    Legal text has highly variable sentence lengths. Word boundaries
    are more semantically meaningful than character counts for this domain.

    chunk_size=500 words fits comfortably in most LLM context windows
    while being specific enough for precise retrieval.

    overlap=50 words ensures clauses at chunk boundaries appear
    fully in at least one chunk — critical for legal accuracy.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if len(chunk.strip()) > 50:  # skip near-empty chunks
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap) words
        # This creates the overlap between consecutive chunks
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
                  doc_type: str, raw_content: str) -> int:
    """
    Inserts a document into legal_documents table.
    Returns the generated document ID for linking chunks.
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

    if result:
        return result[0]
    return None


def save_chunks(conn, document_id: int, chunks: list[str],
                domain: str, doc_type: str, source_url: str) -> int:
    """
    Inserts all chunks for a document into document_chunks table.
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
    """Marks a document as fully processed in legal_documents."""
    cur = conn.cursor()
    cur.execute(
        "UPDATE legal_documents SET is_processed = TRUE WHERE id = %s;",
        (document_id,)
    )
    conn.commit()
    cur.close()


def process_messages(messages: list[dict]) -> dict:
    """
    Core processing function.

    Takes a list of raw Kafka messages and for each one:
    1. Cleans the raw text
    2. Chunks it into 500-word overlapping segments
    3. Saves the document + chunks to PostgreSQL

    Multi-part documents (Companies Act was split into 2 Kafka messages)
    are reassembled here before processing.
    """
    # Reassemble multi-part documents
    # Group messages by source_url, sort by part_index, concatenate content
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

    results = {
        "documents_processed": 0,
        "total_chunks": 0,
        "failed": []
    }

    conn = get_db_connection()

    for url, doc_data in doc_parts.items():
        try:
            # Sort parts by index and join
            sorted_parts = [
                doc_data["parts"][i]
                for i in sorted(doc_data["parts"].keys())
            ]
            full_content = "\n".join(sorted_parts)

            logger.info(f"Processing: {doc_data['title']}")
            logger.info(f"  Raw length: {len(full_content):,} chars")

            # Step 1: Clean
            cleaned = clean_text(full_content)
            logger.info(f"  Cleaned length: {len(cleaned):,} chars")

            # Step 2: Chunk
            chunks = chunk_text(cleaned, chunk_size=500, overlap=50)
            logger.info(f"  Chunks created: {len(chunks)}")

            # Step 3: Save document
            doc_id = save_document(
                conn,
                title=doc_data["title"],
                source_url=doc_data["source_url"],
                domain=doc_data["domain"],
                doc_type=doc_data["doc_type"],
                raw_content=cleaned
            )

            if not doc_id:
                logger.warning(f"  Document already exists, skipping: {doc_data['title']}")
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

            # Step 5: Mark as processed
            mark_document_processed(conn, doc_id)

            logger.info(f"  ✅ Done — {saved} chunks saved to PostgreSQL")
            results["documents_processed"] += 1
            results["total_chunks"] += saved

        except Exception as e:
            logger.error(f"  ❌ Failed: {doc_data['title']} — {e}")
            results["failed"].append(doc_data["title"])

    conn.close()
    return results


if __name__ == "__main__":
    """
    In production this processor runs as a Spark Structured Streaming job
    consuming directly from Kafka. For now we run it in batch mode —
    consuming all available messages from the topic in one shot.

    The Airflow DAG will orchestrate this to run on a schedule.
    """
    from kafka import KafkaConsumer

    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    logger.info("=" * 60)
    logger.info("Sheria Intelligence — Spark Processor")
    logger.info(f"Consuming from topic: {topic}")
    logger.info("=" * 60)

    # Consume all messages currently in the topic
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",       # start from the beginning
        enable_auto_commit=True,
        group_id="sheria-processor",
        consumer_timeout_ms=5000            # stop after 5s of no new messages
    )

    messages = []
    for msg in consumer:
        messages.append(msg.value)

    consumer.close()
    logger.info(f"Consumed {len(messages)} messages from Kafka")

    if not messages:
        logger.warning("No messages found in topic. Run kafka_producer.py first.")
        exit(0)

    # Process all messages
    results = process_messages(messages)

    logger.info("\n" + "=" * 60)
    logger.info("📊 Processing Summary")
    logger.info("=" * 60)
    logger.info(f"   Documents processed: {results['documents_processed']}")
    logger.info(f"   Total chunks saved:  {results['total_chunks']}")
    logger.info(f"   Failed:              {len(results['failed'])}")
    if results["failed"]:
        logger.info(f"   Failed docs: {results['failed']}")
    logger.info("=" * 60)
