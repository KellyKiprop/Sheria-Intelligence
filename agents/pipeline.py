import os
import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from research_agent import SheriaState, research_agent
from analysis_agent import analysis_agent
from citation_agent import citation_agent
from drafting_agent import drafting_agent

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Retry counter ─────────────────────────────────────────────
# We track retries in state to prevent infinite loops.
# If the Citation Agent keeps failing after MAX_RETRIES,
# we pass whatever we have to the Drafting Agent anyway.
MAX_RETRIES = 2


def research_with_broader_scope(state: SheriaState) -> SheriaState:
    """
    Retry node — called when Citation Agent sets needs_retry=True.

    On retry we ignore the detected domain and search across
    all domains with a higher top_k. This gives the Analysis
    Agent more material to work with on the second pass.

    We also increment a retry counter so we don't loop forever.
    """
    logger.info("🔄 Retry — broadening search scope")

    retry_count = state.get("retry_count", 0) + 1

    # Force domain to None — search everything
    broader_state = {
        **state,
        "domain": None,
        "chunks": [],
        "analysis": None,
        "citations": None,
        "needs_retry": False,
        "retry_count": retry_count
    }

    return research_agent(broader_state)


def should_retry(state: SheriaState) -> Literal["retry", "draft"]:
    """
    Conditional edge function — LangGraph calls this after
    the Citation Agent to decide which node to go to next.

    Returns "retry" → goes to research_with_broader_scope node
    Returns "draft" → goes to drafting_agent node

    We cap retries at MAX_RETRIES to prevent infinite loops.
    """
    needs_retry = state.get("needs_retry", False)
    retry_count = state.get("retry_count", 0)

    if needs_retry and retry_count < MAX_RETRIES:
        logger.info(f"Routing to retry (attempt {retry_count + 1}/{MAX_RETRIES})")
        return "retry"
    else:
        if retry_count >= MAX_RETRIES:
            logger.warning(f"Max retries reached — proceeding to drafting anyway")
        return "draft"


def build_pipeline() -> StateGraph:
    """
    Builds and compiles the LangGraph pipeline.

    The graph has 5 nodes:
    - research:  Research Agent
    - analysis:  Analysis Agent
    - citation:  Citation Agent
    - retry:     Broader Research (on citation failure)
    - draft:     Drafting Agent

    And these edges:
    research → analysis → citation → [conditional]
                                         ↓ retry → analysis → citation → [conditional]
                                         ↓ draft → END

    The retry loop can run MAX_RETRIES times before
    forcing through to the Drafting Agent.
    """
    graph = StateGraph(SheriaState)

    # Add nodes
    graph.add_node("research", research_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("citation", citation_agent)
    graph.add_node("retry", research_with_broader_scope)
    graph.add_node("draft", drafting_agent)

    # Fixed edges — always go in this direction
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "citation")
    graph.add_edge("retry", "analysis")    # after retry, re-analyze
    graph.add_edge("draft", END)

    # Conditional edge — after citation, decide retry or draft
    graph.add_conditional_edges(
        "citation",
        should_retry,
        {
            "retry": "retry",
            "draft": "draft"
        }
    )

    # Entry point
    graph.set_entry_point("research")

    return graph.compile()


def run_pipeline(query: str, user_tier: str = "public") -> dict:
    """
    Main entry point for running the full Sheria pipeline.

    Takes a query and user tier, runs the full agent chain,
    and returns the final state with response, citations,
    and metadata.
    """
    logger.info("=" * 65)
    logger.info("Sheria Intelligence Pipeline — Starting")
    logger.info(f"Query: {query}")
    logger.info(f"Tier:  {user_tier}")
    logger.info("=" * 65)

    pipeline = build_pipeline()

    initial_state: SheriaState = {
        "query": query,
        "domain": None,
        "user_tier": user_tier,
        "chunks": [],
        "analysis": None,
        "citations": None,
        "response": None,
        "needs_retry": False,
        "retry_count": 0
    }

    final_state = pipeline.invoke(initial_state)

    logger.info("Pipeline complete")
    return final_state


if __name__ == "__main__":
    test_cases = [
        {
            "query": "Can my employer terminate my contract without notice?",
            "user_tier": "public"
        },
        {
            "query": "What are the requirements to register a company in Kenya?",
            "user_tier": "professional"
        },
        {
            "query": "What happens if my landlord evicts me without a court order?",
            "user_tier": "public"
        }
    ]

    for test in test_cases:
        result = run_pipeline(test["query"], test["user_tier"])

        print(f"\n{'='*65}")
        print(f"FINAL RESPONSE — {test['user_tier'].upper()} TIER")
        print('='*65)
        print(result["response"])

        print(f"\n📚 Sources used:")
        for c in (result["citations"] or []):
            print(f"  - {c['title']}")
            print(f"    {c['source_url']}")

        print(f"\n📊 Pipeline metadata:")
        print(f"  Domain:      {result['domain']}")
        print(f"  Chunks used: {len(result['chunks'])}")
        print(f"  Retries:     {result.get('retry_count', 0)}")
        print('='*65)
