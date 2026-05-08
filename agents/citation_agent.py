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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama-3.3-70b-versatile"


def build_citation_prompt(analysis: str, chunks: list[dict]) -> str:
    """
    Builds the prompt for citation verification.

    We give the model the analysis output and all retrieved sources,
    then ask it to check every legal claim against those sources.
    The model returns a structured verdict for each claim.
    """
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

Respond in this exact format:

VERIFICATION RESULT: PASS or FAIL

CLAIMS VERIFIED:
- [Claim 1]: SUPPORTED by [Source N] / UNSUPPORTED
- [Claim 2]: SUPPORTED by [Source N] / UNSUPPORTED
(continue for all claims)

UNSUPPORTED CLAIMS:
List any claims not traceable to sources, or write "None"

CONFIDENCE: HIGH / MEDIUM / LOW
(HIGH = all claims supported, MEDIUM = minor gaps, LOW = major unsupported claims)

RECOMMENDATION:
One sentence on whether this analysis is safe to deliver to the user.
"""


def parse_citation_result(verification_text: str) -> dict:
    """
    Parses the citation agent's structured output into a dict.

    We extract:
    - passed: whether verification passed overall
    - unsupported_claims: list of unsupported claims found
    - confidence: HIGH / MEDIUM / LOW
    - recommendation: the model's recommendation sentence
    - raw: full verification text for logging
    """
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
    """
    Builds a clean citations list from the retrieved chunks.
    This is what gets attached to the final response so the user
    can see exactly which laws and sections were used.
    """
    citations = []
    seen_titles = set()

    for i, chunk in enumerate(chunks, 1):
        title = chunk["title"]
        if title not in seen_titles:
            citations.append({
                "source_number": i,
                "title": title,
                "domain": chunk["domain"],
                "source_url": chunk["source_url"],
                "similarity": chunk["similarity"]
            })
            seen_titles.add(title)

    return citations


def citation_agent(state: SheriaState) -> SheriaState:
    """
    Citation Agent — Node 3 in the LangGraph pipeline.

    Responsibilities:
    1. Read the analysis produced by the Analysis Agent
    2. Verify every legal claim traces back to a retrieved source
    3. Set needs_retry=True if unsupported claims are found
    4. Build a clean citations list for the final response
    5. Return updated state

    This agent is the quality gate of the pipeline.
    A FAIL here means LangGraph routes back to Research Agent
    for a broader retrieval before re-analysis.
    """
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

    # Step 1: Verify claims
    prompt = build_citation_prompt(analysis, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,    # zero temperature — we want deterministic verification
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

    # Step 2: Build citations list
    citations = build_citations_list(chunks)

    # Step 3: Determine if retry needed
    # We retry if verification FAILED and confidence is LOW
    # MEDIUM confidence passes — minor gaps are acceptable
    needs_retry = not result["passed"] and result["confidence"] == "LOW"

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

    print(f"\n{'='*65}")
    print(f"Query: {test['query']}")
    print('='*65)

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

    print(f"\n✅ Verification passed: {not state['needs_retry']}")
    print(f"📚 Citations:")
    for c in state["citations"]:
        print(f"  [{c['source_number']}] {c['title']} ({c['domain']})")
        print(f"       {c['source_url']}")
    print(f"\nNeeds retry: {state['needs_retry']}")
