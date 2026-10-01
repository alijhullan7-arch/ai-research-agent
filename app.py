import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import streamlit as st

from research_agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("Single CrewAI agent • DuckDuckGo search • Groq (openai/gpt-oss-120b)")

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.error(
        "GROQ_API_KEY not found. In Streamlit Cloud open your app > "
        "Settings > Secrets and add: GROQ_API_KEY = \"your_key\""
    )
    st.stop()

topic = st.text_input("Research topic", placeholder="e.g. Solar energy in Pakistan")

if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Researching... this can take 1-2 minutes."):
            try:
                report = run_research(topic.strip(), groq_api_key)
                st.session_state["report"] = report
                st.session_state["topic"] = topic.strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() or "RateLimitError" in str(e):
                    st.error(
                        "Groq's free-tier rate limit was reached. "
                        "Please wait about a minute and click Generate Report again."
                    )
                else:
                    st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    st.download_button(
        label="⬇️ Download report (.md)",
        data=st.session_state["report"],
        file_name=f"{st.session_state['topic'].replace(' ', '_')}_report.md",
        mime="text/markdown",
    )
