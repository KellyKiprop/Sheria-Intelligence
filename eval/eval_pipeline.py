# evals/eval_pipeline.py
import os
import sys
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

sys.path.insert(0, "/home/kelly/Documents/sheria-intelligence/agents")

from pipeline import run_pipeline
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


judge = GroqJudge()

TEST_QUERIES = [
    {
        "id": "employment_public",
        "query": "Can my employer terminate my contract without notice?",
        "user_tier": "public",
    },
    {
        "id": "employment_professional",
        "query": "What are the legal requirements for a valid redundancy process in Kenya?",
        "user_tier": "professional",
    },
    {
        "id": "business_professional",
        "query": "What are the requirements to register a company in Kenya?",
        "user_tier": "professional",
    },
]


def build_test_case(test: dict) -> LLMTestCase:
    logger.info(f"\nRunning pipeline for: [{test['id']}]")
    state = run_pipeline(test["query"], test["user_tier"])

    response = state.get("response") or ""
    chunks = state.get("chunks") or []
    retrieval_context = [c["content"] for c in chunks]

    logger.info(f"  Response length : {len(response)} chars")
    logger.info(f"  Chunks retrieved: {len(retrieval_context)}")

    return LLMTestCase(
        input=test["query"],
        actual_output=response,
        retrieval_context=retrieval_context,
    )


def run_evals():
    test_cases = [build_test_case(t) for t in TEST_QUERIES]

    metrics = [
        AnswerRelevancyMetric(threshold=0.5, model=judge, async_mode=False),
        FaithfulnessMetric(threshold=0.5, model=judge, async_mode=False),
        ContextualRecallMetric(threshold=0.5, model=judge, async_mode=False),
    ]

    logger.info("\nRunning DeepEval evaluation...")
    evaluate(test_cases=test_cases, metrics=metrics)

    print("\n" + "=" * 65)
    print("SHERIA EVAL RESULTS")
    print("=" * 65)

    output = []
    for test, tc in zip(TEST_QUERIES, test_cases):
        print(f"\n[{test['id']}] {test['query']}")
        print(f"  Tier: {test['user_tier']}")
        entry = {
            "id": test["id"],
            "query": test["query"],
            "tier": test["user_tier"],
            "metrics": {}
        }

        for metric in metrics:
            metric.measure(tc)
            name = metric.__class__.__name__.replace("Metric", "")
            status = "PASS" if metric.score >= metric.threshold else "FAIL"
            print(f"  {name:<22} {metric.score:.2f}  {status}")
            if metric.reason:
                reason = metric.reason[:120] + "..." if len(metric.reason) > 120 else metric.reason
                print(f"    Reason: {reason}")
            entry["metrics"][metric.__class__.__name__] = {
                "score": round(metric.score, 4),
                "passed": metric.score >= metric.threshold,
                "reason": metric.reason,
            }

        output.append(entry)

    print("\n" + "=" * 65)

    out_path = "/home/kelly/Documents/sheria-intelligence/evals/results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_evals()
