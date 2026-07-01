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

FOLLOWUP_DELIMITER = "---FOLLOWUPS---"

# How many prior turns we inject into the prompt.
# Kept small on purpose — legal Q&A doesn't need deep history,
# just enough for "what if it's a first offence"-style follow-ups
# to resolve correctly. Larger windows dilute the prompt and
# inflate token usage for no real accuracy gain.
MAX_HISTORY_TURNS = 3


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


def format_conversation_history(history: list[dict]) -> str:
    """
    Formats prior conversation turns into a compact transcript
    block for prompt injection.

    Expects each turn as {"query": str, "response": str}.
    Only the last MAX_HISTORY_TURNS are used — older turns are
    dropped silently, which is the right tradeoff for a stateless
    per-request LLM call where token budget matters more than
    perfect long-term recall.
    """
    if not history:
        return ""

    recent = history[-MAX_HISTORY_TURNS:]
    lines = ["PRIOR CONVERSATION (for context only — do not re-answer these):"]
    for turn in recent:
        q = turn.get("query", "").strip()
        r = turn.get("response", "").strip()
        if not q:
            continue
        r_short = (r[:300] + "…") if len(r) > 300 else r
        lines.append(f"User asked: {q}")
        lines.append(f"You answered: {r_short}")
    lines.append("")
    return "\n".join(lines)


def build_drafting_prompt(
    query: str,
    analysis: str,
    citations_text: str,
    user_tier: str,
    conversation_history: list[dict] | None = None
) -> str:
    """
    Builds the final drafting prompt.

    The Drafting Agent's job is not to re-analyze — the Analysis Agent
    already did that. Its job is to take the analysis and rewrite it
    into the appropriate format and tone for the user tier, while
    staying coherent with any prior turns in the conversation.
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

    history_block = format_conversation_history(conversation_history or [])
    history_instruction = (
        "If the user's current query references or builds on the prior "
        "conversation (e.g. 'what about...', 'and if...', pronouns like "
        "'it' or 'that'), answer it as a natural continuation — don't "
        "ask them to repeat context they've already given.\n\n"
        if history_block else ""
    )

    return f"""You are the final drafting agent for the Sheria Intelligence Platform.
Your job is to take the legal analysis below and rewrite it into a
polished response appropriate for the user tier.

{history_block}{history_instruction}USER QUERY: {query}
USER TIER: {user_tier}

LEGAL ANALYSIS TO REWRITE:
{analysis}

CITATIONS TO INCLUDE:
{citations_text}

{format_instruction}

Important: Do not add any legal claims beyond what is in the analysis above.
Your job is presentation and clarity, not new analysis.

After writing the full response, add the exact delimiter line
{FOLLOWUP_DELIMITER}
followed by exactly 3 short follow-up questions a person might
naturally ask next, one per line, no numbering, no bullets — just
the plain question text. These should be genuinely useful next
questions a Kenyan citizen would want to ask given this topic,
not generic filler.
"""


def parse_response_and_followups(raw_text: str) -> tuple[str, list[str]]:
    """
    Splits the LLM output into the main response and the follow-up
    question list, using the delimiter the prompt asked for.

    Defensive by design: if the model doesn't include the delimiter
    (it happens), we just return the full text as the response and
    an empty follow-ups list rather than crashing.
    """
    if FOLLOWUP_DELIMITER not in raw_text:
        return raw_text.strip(), []

    main_part, _, followup_part = raw_text.partition(FOLLOWUP_DELIMITER)

    followups = [
        line.strip().lstrip("-•0123456789. ").strip()
        for line in followup_part.strip().split("\n")
        if line.strip()
    ]
    followups = followups[:3]

    return main_part.strip(), followups


def drafting_agent(state: SheriaState) -> SheriaState:
    """
    Drafting Agent — Final node in the LangGraph pipeline.

    Responsibilities:
    1. Take the verified analysis from Citation Agent
    2. Format citations into a readable reference section
    3. Inject prior conversation turns for continuity
    4. Call Groq to produce the final polished response + follow-ups
    5. Return the completed state with response and follow_up_questions filled in

    This is what the user sees. Everything upstream exists
    to make this output accurate and trustworthy.
    """
    query = state["query"]
    analysis = state["analysis"]
    citations = state["citations"] or []
    user_tier = state["user_tier"]
    conversation_history = state.get("conversation_history") or []

    logger.info(f"Drafting Agent — producing final response for: '{query}'")
    logger.info(f"User tier: {user_tier} | History turns: {len(conversation_history)}")

    if not analysis:
        return {
            **state,
            "response": (
                "I was unable to generate a response for this query. "
                "Please try rephrasing your question or consult a qualified advocate."
            ),
            "follow_up_questions": []
        }

    citations_text = format_citations_for_response(citations)
    prompt = build_drafting_prompt(
        query, analysis, citations_text, user_tier, conversation_history
    )

    response = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )

    raw_text = response.choices[0].message.content
    final_response, follow_up_questions = parse_response_and_followups(raw_text)

    logger.info(
        f"Response drafted — {len(final_response)} chars, "
        f"{len(follow_up_questions)} follow-ups"
    )

    return {
        **state,
        "response": final_response,
        "follow_up_questions": follow_up_questions
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
            "needs_retry": False,
            "conversation_history": [],
            "follow_up_questions": None
        }

        state = research_agent(initial_state)
        state = analysis_agent(state)
        state = citation_agent(state)
        state = drafting_agent(state)

        print(f"\n{state['response']}")
        print(f"\nFollow-ups: {state.get('follow_up_questions')}")
        print(f"\n{'='*65}")
