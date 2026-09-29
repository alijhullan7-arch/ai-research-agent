import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from ddgs import DDGS

# --- Workaround for a known CrewAI bug ---
# CrewAI tags messages with an internal "cache_breakpoint" key meant only for
# Anthropic models. Groq (and other non-Anthropic providers) reject it with:
# "property 'cache_breakpoint' is unsupported". This patch makes that marker
# a no-op so the key is never added. Safe to remove once CrewAI ships an
# official fix.
try:
    import crewai.llms.cache as _crewai_cache

    def _noop_mark_cache_breakpoint(message):
        return message

    _crewai_cache.mark_cache_breakpoint = _noop_mark_cache_breakpoint

    import crewai.agents.crew_agent_executor as _crew_agent_executor
    _crew_agent_executor.mark_cache_breakpoint = _noop_mark_cache_breakpoint
except Exception:
    pass  # if CrewAI's internals changed, fail silently and let the app run


# ---------- 1. Search tool (DuckDuckGo, free, no API key) ----------
@tool("DuckDuckGo Search")
def search_web(query: str) -> str:
    """Search the internet with DuckDuckGo. Input is a short search query.
    Returns titles, links and short snippets of the top results."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        link = r.get("href", "")
        body = r.get("body", "")[:300]  # keep it short (Groq free tier has token limits)
        lines.append(f"- {title}\n  {link}\n  {body}")
    return "\n".join(lines)


# ---------- 2. Main function called by the Streamlit app ----------
def run_research(topic: str, groq_api_key: str) -> str:
    os.environ["GROQ_API_KEY"] = groq_api_key

    # "groq/" prefix tells CrewAI (via LiteLLM) to use Groq
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.3,
        max_tokens=4000,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research the topic '{topic}' using web search and write a clear, accurate report.",
        backstory=(
            "You are an experienced research analyst. You search the web, "
            "compare sources, and write well-structured reports. "
            "You never invent facts and always mention your sources."
        ),
        tools=[search_web],
        llm=llm,
        max_iter=6,          # limit search/think loops
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: {topic}\n"
            "Use the search tool 3 to 5 times with different queries. "
            "Then write a detailed report based only on what you found."
        ),
        expected_output=(
            "A Markdown report with these sections: Title, Introduction, "
            "Key Findings (bullet points), Detailed Analysis, Conclusion, "
            "and a Sources list with URLs."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"topic": topic})
    return result.raw
