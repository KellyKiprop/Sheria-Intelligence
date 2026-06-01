import os
from groq import Groq
import logging
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

def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a structured context block.
    Each chunk is labelled with its source so the model
    knows exactly where each legal provision comes from.
    """
    if not chunks:
        return "No relevant legal provisions found."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(
            f"[Source {i}: {chunk['title']} | "
            f"Domain: {chunk['domain']} | "
            f"Similarity: {chunk['similarity']}]\n"
            f"{chunk['content']}"
        )

    return "\n\n---\n\n".join(formatted)


def build_analysis_prompt(query: str, chunks_text: str, user_tier: str) -> str:
    """
    Builds the prompt for the Analysis Agent.

    Two modes based on user tier:
    - public: plain language, practical guidance, no jargon
    - professional: technical analysis, statutory references, edge cases
    """
    if user_tier == "professional":
        depth_instruction = """
Provide a detailed technical legal analysis including:
- Specific statutory provisions and their exact wording
- Cross-references between relevant sections
- Legal implications and potential edge cases
- Relevant legal principles that apply
"""
    else:
        depth_instruction = """
Provide a clear plain-language explanation that:
- Explains what the law says in simple terms a non-lawyer can understand
- States clearly what rights or obligations apply
- Gives practical guidance on what the person should do
- Avoids excessive legal jargon
"""

    return f"""You are a Kenyan legal analyst for the Sheria Intelligence Platform.
Your role is to analyze legal queries using only the provided Kenyan legal provisions.

IMPORTANT RULES:
- Only use information from the provided legal sources below
- Every legal claim must reference a specific source by its [Source N] label
- If the sources do not contain enough information, say so explicitly
- Never invent or assume legal provisions not present in the sources
- Always clarify this is legal information, not legal advice

USER QUERY:
{query}

RETRIEVED LEGAL PROVISIONS:
{chunks_text}

{depth_instruction}

Structure your response exactly as follows:
1. DIRECT ANSWER — Answer the query in 2-3 sentences
2. LEGAL BASIS — Which laws and sections apply, with source references
3. DETAILED ANALYSIS — Full explanation based on the retrieved provisions
4. PRACTICAL IMPLICATIONS — What this means practically for the person asking
5. LIMITATIONS — What this analysis cannot cover due to gaps in retrieved sources
"""


def analysis_agent(state: SheriaState) -> SheriaState:
    """
    Analysis Agent — Node 2 in the LangGraph pipeline.

    Takes chunks from Research Agent, reasons over them
    using Gemini 2.0 Flash, and returns structured legal analysis.

    Handles empty retrieval gracefully — never hallucinates
    when no relevant provisions are found.
    """
    query = state["query"]
    chunks = state["chunks"]
    user_tier = state["user_tier"]

    logger.info(f"Analysis Agent — {len(chunks)} chunks for: '{query}'")

    if not chunks:
        logger.warning("No chunks — returning insufficient data response")
        return {
            **state,
            "analysis": (
                "I was unable to find relevant Kenyan legal provisions for this query "
                "in the current knowledge base. The relevant law may not yet be indexed. "
                "Please consult a qualified Kenyan advocate for guidance."
            )
        }

    chunks_text = format_chunks_for_prompt(chunks)
    prompt = build_analysis_prompt(query, chunks_text, user_tier)

    logger.info(f"Calling {LLM_MODEL} for legal analysis...")

    response = get_groq_client().chat.completions.create(
    	model=LLM_MODEL,
    	messages=[{"role": "user", "content": prompt}],
    	temperature=0.1,
    	max_tokens=2048,
    )

    analysis = response.choices[0].message.content
    logger.info(f"Analysis complete — {len(analysis)} chars")

    return {
        **state,
        "analysis": analysis
    }


if __name__ == "__main__":
    from research_agent import research_agent

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

        state_after_research = research_agent(initial_state)
        state_after_analysis = analysis_agent(state_after_research)

        print(f"\n{state_after_analysis['analysis']}")
        print(f"\n{'='*65}")
