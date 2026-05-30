"""
Sheria Intelligence Platform — Airflow DAG
Daily ingestion of Kenyan legal documents from Kenya Law.
Schedule: 06:00 EAT (03:00 UTC)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
import logging
import os

logger = logging.getLogger(__name__)

default_args = {
    "owner": "sheria-intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    dag_id="sheria_legal_ingestion_pipeline",
    default_args=default_args,
    description="Daily ingestion of Kenyan legal documents from Kenya Law",
    schedule_interval="0 3 * * *",
    start_date=datetime(2026, 5, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sheria", "legal", "ingestion", "production"],
)


def task_health_check(**context):
    import requests
    import psycopg2
    from kafka import KafkaProducer

    errors = []

    # Check Kenya Law
    try:
        r = requests.get("https://new.kenyalaw.org", timeout=30)
        if r.status_code == 200:
            logger.info("✅ Kenya Law website reachable")
        else:
            logger.warning(f"Kenya Law returned status {r.status_code} - continuing anyway")
    except Exception as e:
        logger.warning(f"Kenya Law check timed out: {e} - continuing anyway")

    # Check Aiven PostgreSQL
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
                "KAFKA_BOOTSTRAP_SERVERS", "sheria_kafka:9092"
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

    logger.info("✅ All health checks passed")


def task_scrape(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from ingestion.scraper import KenyaLawScraper

    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()

    if not docs:
        raise Exception("Scraper returned 0 documents")

    logger.info(f"Scraped {len(docs)} documents")
    context["ti"].xcom_push(key="documents_scraped", value=len(docs))
    return len(docs)


def task_publish_to_kafka(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from ingestion.kafka_producer import KenyaLawScraper, publish_documents

    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()
    results = publish_documents(docs, topic)

    logger.info(f"Published {results['success']}/{len(docs)} documents")
    context["ti"].xcom_push(
        key="messages_published",
        value=results["total_messages"]
    )


def task_process_and_chunk(**context):
    import sys
    import json
    sys.path.insert(0, "/opt/airflow/project")
    from kafka import KafkaConsumer
    from ingestion.spark_processor import process_messages

    topic = os.getenv("KAFKA_TOPIC_RAW_DOCS", "sheria.raw.documents")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "sheria_kafka:9092")

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
        logger.warning("No messages in topic")
        return

    results = process_messages(messages)
    logger.info(f"Processed {results['documents_processed']} docs, {results['total_chunks']} chunks")
    context["ti"].xcom_push(key="chunks_created", value=results["total_chunks"])


def task_embed(**context):
    import sys
    sys.path.insert(0, "/opt/airflow/project")
    from store.embeddings import embed_all_chunks

    results = embed_all_chunks()
    logger.info(f"Embedded {results['embedded']} chunks")
    context["ti"].xcom_push(key="chunks_embedded", value=results["embedded"])


def task_log_summary(**context):
    ti = context["ti"]
    docs = ti.xcom_pull(task_ids="scrape", key="documents_scraped") or 0
    messages = ti.xcom_pull(task_ids="publish_to_kafka", key="messages_published") or 0
    chunks = ti.xcom_pull(task_ids="process_and_chunk", key="chunks_created") or 0
    embedded = ti.xcom_pull(task_ids="embed", key="chunks_embedded") or 0

    logger.info("=" * 60)
    logger.info("Sheria Intelligence — Daily Pipeline Summary")
    logger.info(f"Documents scraped:  {docs}")
    logger.info(f"Kafka messages:     {messages}")
    logger.info(f"Chunks created:     {chunks}")
    logger.info(f"Chunks embedded:    {embedded}")
    logger.info("=" * 60)


def task_handle_failure(**context):
    failed_task = context.get("task_instance").task_id
    logger.error(f"❌ Pipeline failed at task: {failed_task}")


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

    handle_failure = PythonOperator(
        task_id="handle_failure",
        python_callable=task_handle_failure,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    start >> health_check >> scrape >> publish_to_kafka
    publish_to_kafka >> process_and_chunk >> embed >> log_summary
    log_summary >> end

    [
        health_check, scrape, publish_to_kafka,
        process_and_chunk, embed, log_summary
    ] >> handle_failure
