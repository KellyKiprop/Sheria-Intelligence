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


def format_citations_for_response(citations: list[dict]) -> str:
    """
    Formats the citations list into a readable reference section
    that appears at the bottom of every response.
    """
    if not citations:
        return "No sources available."

    lines = ["**Legal Sources Referenced:**"]
    for c in citations:
        lines.append(
            f"- {c['title']} | "
            f"{c['source_url']}"
        )

    return "\n".join(lines)


def build_drafting_prompt(
    query: str,
    analysis: str,
    citations_text: str,
    user_tier: str
) -> str:
    """
    Builds the final drafting prompt.

    The Drafting Agent's job is not to re-analyze — the Analysis Agent
    already did that. Its job is to take the analysis and rewrite it
    into the appropriate format and tone for the user tier.

    Public: conversational, warm, empowering
    Professional: formal, structured, precise
    """
    if user_tier == "professional":
        format_instruction = """
Produce a formal legal brief with the following structure:

LEGAL BRIEF — SHERIA INTELLIGENCE PLATFORM
==========================================
RE: [Restate the query as a legal matter]
DATE: [Today's date]

EXECUTIVE SUMMARY
[2-3 sentences summarizing the legal position]

APPLICABLE LAW
[List the relevant Acts and sections]

LEGAL ANALYSIS
[Detailed analysis drawn from the provided analysis]

CONCLUSION
[Clear statement of the legal position]

DISCLAIMER
This brief constitutes legal information only and does not constitute
legal advice. For advice specific to your circumstances, consult a
qualified advocate registered with the Law Society of Kenya.

SOURCES
[Insert citations here]
"""
    else:
        format_instruction = """
Write a warm, clear, helpful response that:
- Opens by directly answering the question in plain language
- Explains the relevant law in simple terms
- Tells the person what they can do next practically
- Uses short paragraphs and simple sentences
- Ends with a brief disclaimer that this is general legal information
- Finishes with the sources section

Tone: helpful, empowering, like a knowledgeable friend explaining the law.
"""

    return f"""You are the final drafting agent for the Sheria Intelligence Platform.
Your job is to take the legal analysis below and rewrite it into a
polished response appropriate for the user tier.

USER QUERY: {query}
USER TIER: {user_tier}

LEGAL ANALYSIS TO REWRITE:
{analysis}

CITATIONS TO INCLUDE:
{citations_text}

{format_instruction}

Important: Do not add any legal claims beyond what is in the analysis above.
Your job is presentation and clarity, not new analysis.
"""


def drafting_agent(state: SheriaState) -> SheriaState:
    """
    Drafting Agent — Final node in the LangGraph pipeline.

    Responsibilities:
    1. Take the verified analysis from Citation Agent
    2. Format citations into a readable reference section
    3. Call Groq to produce the final polished response
    4. Return the completed state with response filled in

    This is what the user sees. Everything upstream exists
    to make this output accurate and trustworthy.
    """
    query = state["query"]
    analysis = state["analysis"]
    citations = state["citations"] or []
    user_tier = state["user_tier"]

    logger.info(f"Drafting Agent — producing final response for: '{query}'")
    logger.info(f"User tier: {user_tier}")

    if not analysis:
        return {
            **state,
            "response": (
                "I was unable to generate a response for this query. "
                "Please try rephrasing your question or consult a qualified advocate."
            )
        }

    citations_text = format_citations_for_response(citations)
    prompt = build_drafting_prompt(query, analysis, citations_text, user_tier)

    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,    # slightly higher than analysis — we want readable prose
        max_tokens=2048,
    )

    final_response = response.choices[0].message.content
    logger.info(f"Response drafted — {len(final_response)} chars")

    return {
        **state,
        "response": final_response
    }


if __name__ == "__main__":
    from research_agent import research_agent
    from analysis_agent import analysis_agent
    from citation_agent import citation_agent

    test_queries = [
        {
            "query": "Can my employer terminate my contract without notice?",
            "user_tier": "public"
        },
        {
            "query": "What are the requirements to register a company in Kenya?",
            "user_tier": "professional"
        }
    ]

    for test in test_queries:
        print(f"\n{'='*65}")
        print(f"Query: {test['query']}")
        print(f"Tier:  {test['user_tier']}")
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
        state = drafting_agent(state)

        print(f"\n{state['response']}")
        print(f"\n{'='*65}")
