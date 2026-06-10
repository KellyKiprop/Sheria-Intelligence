import os
import logging
from groq import Groq
from dotenv import load_dotenv
from research_agent import SheriaState

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

LLM_MODEL = "llama-3.3-70b-versatile"

def build_citation_prompt(analysis: str, chunks: list[dict]) -> str:
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        sources_text += (
            f"[Source {i}: {chunk['title']} | {chunk['domain']}]\n"
            f"{chunk['content']}\n\n"
        )
    return f"""You are a legal citation verifier for the Sheria Intelligence Platform.
Your job is to verify that every legal claim in the analysis below is supported
by the provided source documents.

ANALYSIS TO VERIFY:
{analysis}

SOURCE DOCUMENTS:
{sources_text}

VERIFICATION TASK:
1. Identify every legal claim made in the analysis
2. Check whether each claim is supported by the source documents
3. Flag any claim that cannot be traced to a specific source

Be LENIENT — general statements about what a law covers are acceptable.
Only flag claims that directly contradict or have no relation to the sources.

Respond in this exact format:
VERIFICATION RESULT: PASS or FAIL
CLAIMS VERIFIED:
- [Claim 1]: SUPPORTED by [Source N] / UNSUPPORTED
- [Claim 2]: SUPPORTED by [Source N] / UNSUPPORTED
UNSUPPORTED CLAIMS:
List any claims not traceable to sources, or write "None"
CONFIDENCE: HIGH / MEDIUM / LOW
RECOMMENDATION:
One sentence on whether this analysis is safe to deliver to the user.
"""

def parse_citation_result(verification_text: str) -> dict:
    lines = verification_text.strip().split("\n")
    passed = True
    unsupported_claims = []
    confidence = "HIGH"
    recommendation = ""
    in_unsupported_section = False

    for line in lines:
        line = line.strip()
        if line.startswith("VERIFICATION RESULT:"):
            result = line.replace("VERIFICATION RESULT:", "").strip()
            passed = result.upper() == "PASS"
        elif line.startswith("UNSUPPORTED CLAIMS:"):
            in_unsupported_section = True
        elif in_unsupported_section:
            if line.startswith("CONFIDENCE:"):
                in_unsupported_section = False
                confidence = line.replace("CONFIDENCE:", "").strip()
            elif line and line != "None" and not line.startswith("CLAIMS"):
                unsupported_claims.append(line)
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip()
        elif line.startswith("RECOMMENDATION:"):
            recommendation = line.replace("RECOMMENDATION:", "").strip()

    return {
        "passed": passed,
        "unsupported_claims": unsupported_claims,
        "confidence": confidence,
        "recommendation": recommendation,
        "raw": verification_text
    }

def build_citations_list(chunks: list[dict]) -> list[dict]:
    citations = []
    seen_titles = set()
    for i, chunk in enumerate(chunks, 1):
        title = chunk["title"]
        if title not in seen_titles:
            citations.append({
                "source_number": i,
                "title": title,
                "domain": chunk["domain"],
                "source_url": chunk.get("source_url") or "",
                "similarity": chunk["similarity"]
            })
            seen_titles.add(title)
    return citations

def citation_agent(state: SheriaState) -> SheriaState:
    analysis = state["analysis"]
    chunks = state["chunks"]
    query = state["query"]

    logger.info(f"Citation Agent — verifying analysis for: '{query}'")

    if not chunks or not analysis:
        logger.warning("No chunks or analysis to verify")
        return {
            **state,
            "citations": [],
            "needs_retry": False
        }

    prompt = build_citation_prompt(analysis, chunks)
    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    verification_text = response.choices[0].message.content
    result = parse_citation_result(verification_text)

    logger.info(f"Verification result: {'PASS' if result['passed'] else 'FAIL'}")
    logger.info(f"Confidence: {result['confidence']}")

    if result["unsupported_claims"]:
        logger.warning(f"Unsupported claims found: {len(result['unsupported_claims'])}")
        for claim in result["unsupported_claims"]:
            logger.warning(f"  → {claim}")

    citations = build_citations_list(chunks)

    # Only retry on major failures — more than 2 unsupported claims AND low confidence
    needs_retry = (
        not result["passed"]
        and result["confidence"] == "LOW"
        and len(result["unsupported_claims"]) > 2
    )

    if needs_retry:
        logger.warning("Confidence LOW — flagging for retry")
    else:
        logger.info("Citation check passed — proceeding to Drafting Agent")

    return {
        **state,
        "citations": citations,
        "needs_retry": needs_retry
    }

if __name__ == "__main__":
    from research_agent import research_agent
    from analysis_agent import analysis_agent

    test = {
        "query": "Can my employer terminate my contract without notice?",
        "user_tier": "public"
    }

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

    state = research_agent(initial_state)
    state = analysis_agent(state)
    state = citation_agent(state)

    print(f"\nVerification passed: {not state['needs_retry']}")
    print(f"Citations:")
    for c in state["citations"]:
        print(f"  [{c['source_number']}] {c['title']} ({c['domain']})")
        print(f"       {c['source_url']}")
    print(f"\nNeeds retry: {state['needs_retry']}")
