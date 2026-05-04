import json
import logging
import time
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os
from dotenv import load_dotenv
from scraper import KenyaLawScraper, LegalDocument

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Kafka's default max message size is 1MB
# We stay safely under at 900KB to account for message metadata overhead
MAX_CONTENT_BYTES = 900_000


def create_producer() -> KafkaProducer:
    """
    Creates a Kafka producer with JSON serialization.
    acks='all' means the broker waits for all replicas to confirm
    before acknowledging — no silent message loss.
    """
    return KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=3,
        retry_backoff_ms=500,
    )


def split_document(doc: LegalDocument) -> list[str]:
    """
    Splits a large document into parts that fit within Kafka's message limit.

    We split at paragraph boundaries (newlines) rather than at arbitrary
    character positions — this preserves the legal text structure and ensures
    no sentence or clause gets cut in half mid-message.

    Returns a list of text parts, each under MAX_CONTENT_BYTES.
    """
    content_bytes = doc.raw_content.encode("utf-8")

    # Small enough — no splitting needed
    if len(content_bytes) <= MAX_CONTENT_BYTES:
        return [doc.raw_content]

    logger.info(
        f"⚠️  Large document ({len(content_bytes):,} bytes) — "
        f"splitting at paragraph boundaries: {doc.title}"
    )

    paragraphs = doc.raw_content.split("\n")
    parts = []
    current_paragraphs = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_size = len(paragraph.encode("utf-8"))

        if current_size + paragraph_size > MAX_CONTENT_BYTES:
            # Current batch is full — save it and start a new one
            if current_paragraphs:
                parts.append("\n".join(current_paragraphs))
            current_paragraphs = [paragraph]
            current_size = paragraph_size
        else:
            current_paragraphs.append(paragraph)
            current_size += paragraph_size

    # Don't forget the last batch
    if current_paragraphs:
        parts.append("\n".join(current_paragraphs))

    logger.info(f"   → Split into {len(parts)} parts")
    return parts


def build_message(doc: LegalDocument, content: str, part_index: int, total_parts: int) -> dict:
    """
    Builds the Kafka message payload for a single document part.

    We include metadata like part_index and total_parts so the downstream
    Spark consumer knows how to reassemble multi-part documents correctly.
    """
    return {
        "title": doc.title,
        "source_url": doc.source_url,
        "domain": doc.domain,
        "doc_type": doc.doc_type,
        "version_date": doc.version_date,
        "raw_content": content,
        "part_index": part_index,
        "total_parts": total_parts,
        "content_length": len(content),
        "ingested_at": datetime.utcnow().isoformat(),
    }


def publish_documents(docs: list[LegalDocument], topic: str) -> dict:
    """
    Main publishing function.

    For each document:
    1. Check if it exceeds Kafka's message size limit
    2. If yes — split at paragraph boundaries into safe-sized parts
    3. Publish each part as a separate Kafka message with shared metadata
    4. Track success/failure per document (not per part)
    """
    producer = create_producer()
    results = {
        "success": 0,
        "failed": 0,
        "total_messages": 0,
        "failed_titles": []
    }

    for doc in docs:
        try:
            parts = split_document(doc)
            total_parts = len(parts)
            doc_failed = False

            for i, part in enumerate(parts):
                message = build_message(doc, part, i, total_parts)

                # Use source_url + part index as the Kafka message key
                # This ensures all parts of the same document go to the
                # same partition — preserving order for the consumer
                message_key = f"{doc.source_url}__part{i}"

                future = producer.send(
                    topic=topic,
                    key=message_key,
                    value=message
                )

                # Block until broker confirms receipt
                record_metadata = future.get(timeout=10)
                results["total_messages"] += 1

                logger.info(
                    f"✅ {doc.title} | part {i+1}/{total_parts} | "
                    f"partition={record_metadata.partition} | "
                    f"offset={record_metadata.offset} | "
                    f"{len(part):,} chars"
                )

            if not doc_failed:
                results["success"] += 1

        except KafkaError as e:
            logger.error(f"❌ Failed to publish {doc.title}: {e}")
            results["failed"] += 1
            results["failed_titles"].append(doc.title)

    producer.flush()
    producer.close()
    return results


if __name__ == "__main__":
    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")

    logger.info("=" * 60)
    logger.info("Sheria Intelligence — Ingestion Pipeline")
    logger.info(f"Target topic: {topic}")
    logger.info("=" * 60)

    # Step 1: Scrape all target Acts from Kenya Law
    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()

    if not docs:
        logger.error("No documents scraped. Aborting.")
        exit(1)

    logger.info(f"\nScrape complete — {len(docs)} documents ready for publishing")

    # Step 2: Publish to Kafka
    logger.info(f"Publishing to Kafka topic: {topic}\n")
    results = publish_documents(docs, topic)

    # Step 3: Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Pipeline Run Summary")
    logger.info("=" * 60)
    logger.info(f"   Documents scraped:    {len(docs)}")
    logger.info(f"   Documents published:  {results['success']}")
    logger.info(f"   Documents failed:     {results['failed']}")
    logger.info(f"   Total Kafka messages: {results['total_messages']}")
    if results["failed_titles"]:
        logger.info(f"   Failed: {results['failed_titles']}")
    logger.info("=" * 60)
