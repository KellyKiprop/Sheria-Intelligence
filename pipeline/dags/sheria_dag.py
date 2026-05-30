"""
Sheria Intelligence Platform — Airflow DAG
==========================================
Orchestrates the full legal document ingestion pipeline:

    Scrape Kenya Law → Publish to Kafka → Process & Chunk → Embed

Schedule: Daily at 06:00 EAT (East Africa Time / UTC+3)
Owner:    Sheria Intelligence Platform
Contact:  Data Engineering Team

This DAG is designed to be readable and deployable by Kenya Law's
ICT team on their own infrastructure with minimal modification.
All configuration is driven by environment variables.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, "/home/kelly/Documents/sheria-intelligence")

logger = logging.getLogger(__name__)

# ── Default arguments ─────────────────────────────────────────
# These apply to every task in the DAG unless overridden.
# retries=2 means each task retries twice before failing.
# retry_delay=5min gives the external service time to recover.

default_args = {
    "owner": "sheria-intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ── DAG definition ────────────────────────────────────────────

dag = DAG(
    dag_id="sheria_legal_ingestion_pipeline",
    default_args=default_args,
    description="Daily ingestion of Kenyan legal documents from Kenya Law",
    # 06:00 EAT = 03:00 UTC (EAT is UTC+3)
    schedule_interval="0 3 * * *",
    start_date=datetime(2026, 5, 1),
    catchup=False,                  # don't backfill missed runs
    max_active_runs=1,              # only one run at a time
    tags=["sheria", "legal", "ingestion", "production"],
    doc_md="""
## Sheria Intelligence — Legal Ingestion Pipeline

### What this DAG does
Runs every morning at 06:00 EAT to keep the Sheria knowledge base
current with the latest Kenyan legal documents from Kenya Law.

### Pipeline stages
1. **health_check** — Verifies Kenya Law site and database are reachable
2. **scrape** — Pulls Acts from new.kenyalaw.org
3. **publish_to_kafka** — Streams scraped documents into Kafka
4. **process_and_chunk** — Cleans and chunks documents via Spark processor
5. **embed** — Generates vector embeddings and stores in pgvector
6. **log_summary** — Logs pipeline metrics for monitoring

### Configuration
All configuration is via environment variables. See `.env.example`.

### Failure handling
Each task retries twice with a 5-minute delay.
On final failure, check Airflow logs for the specific error.

### Contacts
Built by: Kelly Kiprop — Data Engineer
Platform: Sheria Intelligence
    """,
)


# ── Task functions ────────────────────────────────────────────

def task_health_check(**context):
    """
    Task 1: Health Check

    Before scraping, verify:
    - Kenya Law website is reachable
    - Aiven PostgreSQL is reachable
    - Kafka is reachable

    If any check fails, the DAG stops early rather than
    wasting time scraping into a broken pipeline.
    """
    import requests
    import psycopg2
    from kafka import KafkaProducer
    from dotenv import load_dotenv

    load_dotenv("/home/kelly/Documents/sheria-intelligence/.env")

    errors = []

    # Check Kenya Law
    try:
        r = requests.get(
            "https://new.kenyalaw.org",
            timeout=15
        )
        if r.status_code == 200:
            logger.info("✅ Kenya Law website reachable")
        else:
            errors.append(f"Kenya Law returned status {r.status_code}")
    except Exception as e:
        errors.append(f"Kenya Law unreachable: {e}")

    # Check database
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
            sslmode="require"
        )
        conn.close()
        logger.info("✅ Aiven PostgreSQL reachable")
    except Exception as e:
        errors.append(f"Database unreachable: {e}")

    # Check Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
            )
        )
        producer.close()
        logger.info("✅ Kafka reachable")
    except Exception as e:
        errors.append(f"Kafka unreachable: {e}")

    if errors:
        raise Exception(
            f"Health check failed — {len(errors)} error(s):\n" +
            "\n".join(errors)
        )

    logger.info("✅ All health checks passed — proceeding with pipeline")


def task_scrape(**context):
    """
    Task 2: Scrape Kenya Law

    Pulls all target Acts from new.kenyalaw.org.
    Stores scraped document count in XCom so downstream
    tasks can log and verify the expected volume.

    XCom is Airflow's mechanism for passing small pieces
    of data between tasks — think of it as a shared notepad.
    """
    from ingestion.scraper import KenyaLawScraper

    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()

    if not docs:
        raise Exception("Scraper returned 0 documents — aborting pipeline")

    logger.info(f"Scraped {len(docs)} documents successfully")

    # Push to XCom for downstream tasks to read
    context["ti"].xcom_push(key="documents_scraped", value=len(docs))

    return len(docs)


def task_publish_to_kafka(**context):
    """
    Task 3: Publish to Kafka

    Re-runs the scraper (stateless) and publishes to Kafka.
    We re-scrape rather than passing the full document objects
    through XCom — XCom is for small metadata, not large payloads.

    In a production system with shared storage (S3/GCS),
    we'd save scraped docs to object storage in Task 2
    and read them here. For our architecture, re-scraping
    is clean and adds less than 30 seconds.
    """
    from ingestion.kafka_producer import KenyaLawScraper, publish_documents
    from dotenv import load_dotenv

    load_dotenv("/home/kelly/Documents/sheria-intelligence/.env")

    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()

    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    results = publish_documents(docs, topic)

    if results["failed"] > 0:
        logger.warning(
            f"Some documents failed to publish: {results['failed_titles']}"
        )

    logger.info(
        f"Published {results['success']}/{len(docs)} documents "
        f"({results['total_messages']} Kafka messages)"
    )

    context["ti"].xcom_push(
        key="messages_published",
        value=results["total_messages"]
    )


def task_process_and_chunk(**context):
    """
    Task 4: Process and Chunk

    Consumes messages from Kafka, cleans and chunks each document,
    and stores chunks in PostgreSQL.

    Only processes documents not already in the database —
    idempotent by design. Re-running this task never creates
    duplicate chunks.
    """
    import json
    from kafka import KafkaConsumer
    from ingestion.spark_processor import process_messages
    from dotenv import load_dotenv

    load_dotenv("/home/kelly/Documents/sheria-intelligence/.env")

    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="sheria-airflow-processor",
        consumer_timeout_ms=10000
    )

    messages = [msg.value for msg in consumer]
    consumer.close()

    if not messages:
        logger.warning("No messages in Kafka topic — skipping processing")
        return

    results = process_messages(messages)
    logger.info(
        f"Processed {results['documents_processed']} documents, "
        f"{results['total_chunks']} chunks saved"
    )

    context["ti"].xcom_push(
        key="chunks_created",
        value=results["total_chunks"]
    )


def task_embed(**context):
    """
    Task 5: Generate Embeddings

    Embeds all unembedded chunks using Gemini.
    Idempotent — only embeds chunks with no existing embedding.
    Safe to re-run if interrupted by quota limits.
    """
    from store.embeddings import embed_all_chunks
    from dotenv import load_dotenv

    load_dotenv("/home/kelly/Documents/sheria-intelligence/.env")

    results = embed_all_chunks()
    logger.info(
        f"Embedding complete — "
        f"{results['embedded']} embedded, "
        f"{results['failed']} failed"
    )

    context["ti"].xcom_push(
        key="chunks_embedded",
        value=results["embedded"]
    )


def task_log_summary(**context):
    """
    Task 6: Log Pipeline Summary

    Pulls XCom values from all upstream tasks and logs
    a clean summary of the pipeline run.

    In production this would also:
    - Send a Slack notification to the data team
    - Write metrics to a monitoring table
    - Trigger an alert if numbers look wrong
    """
    ti = context["ti"]

    docs_scraped = ti.xcom_pull(
        task_ids="scrape",
        key="documents_scraped"
    ) or 0

    messages_published = ti.xcom_pull(
        task_ids="publish_to_kafka",
        key="messages_published"
    ) or 0

    chunks_created = ti.xcom_pull(
        task_ids="process_and_chunk",
        key="chunks_created"
    ) or 0

    chunks_embedded = ti.xcom_pull(
        task_ids="embed",
        key="chunks_embedded"
    ) or 0

    execution_date = context["execution_date"]

    logger.info("=" * 65)
    logger.info("Sheria Intelligence — Daily Pipeline Summary")
    logger.info("=" * 65)
    logger.info(f"Run date:           {execution_date}")
    logger.info(f"Documents scraped:  {docs_scraped}")
    logger.info(f"Kafka messages:     {messages_published}")
    logger.info(f"Chunks created:     {chunks_created}")
    logger.info(f"Chunks embedded:    {chunks_embedded}")
    logger.info("=" * 65)


def task_handle_failure(**context):
    """
    Failure callback task — triggered if any upstream task fails.

    In production this sends a Slack/email alert.
    Here it logs a clear failure message with context.
    """
    failed_task = context.get("task_instance").task_id
    execution_date = context.get("execution_date")

    logger.error("=" * 65)
    logger.error("❌ PIPELINE FAILURE")
    logger.error(f"Failed task:    {failed_task}")
    logger.error(f"Execution date: {execution_date}")
    logger.error("Action required: Check Airflow logs for details")
    logger.error("=" * 65)


# ── Task definitions ──────────────────────────────────────────

with dag:

    start = EmptyOperator(task_id="start")

    health_check = PythonOperator(
        task_id="health_check",
        python_callable=task_health_check,
    )

    scrape = PythonOperator(
        task_id="scrape",
        python_callable=task_scrape,
    )

    publish_to_kafka = PythonOperator(
        task_id="publish_to_kafka",
        python_callable=task_publish_to_kafka,
    )

    process_and_chunk = PythonOperator(
        task_id="process_and_chunk",
        python_callable=task_process_and_chunk,
    )

    embed = PythonOperator(
        task_id="embed",
        python_callable=task_embed,
    )

    log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=task_log_summary,
    )

    # Failure handler runs if ANY task fails
    # TriggerRule.ONE_FAILED means this task triggers
    # when at least one upstream task has failed
    handle_failure = PythonOperator(
        task_id="handle_failure",
        python_callable=task_handle_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # ── Task dependencies ─────────────────────────────────────
    # This defines the execution order and branching logic.
    # Read >> as "then run"

    start >> health_check >> scrape >> publish_to_kafka
    publish_to_kafka >> process_and_chunk >> embed >> log_summary
    log_summary >> end

    # Failure handler watches all tasks
    [
        health_check,
        scrape,
        publish_to_kafka,
        process_and_chunk,
        embed,
        log_summary
    ] >> handle_failure
