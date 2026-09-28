# AI Research Agent

CrewAI + DuckDuckGo + Groq (openai/gpt-oss-120b) + Streamlit.

## Deploy (no local run needed)

1. Upload these files to a GitHub repo: app.py, research_agent.py, requirements.txt, .gitignore, README.md
2. Go to https://share.streamlit.io > Create app > select your repo, branch main, main file app.py
3. Advanced settings:
   - Python version: 3.12
   - Secrets: GROQ_API_KEY = "your_groq_key"
4. Click Deploy.

To change the key later: App > Settings > Secrets.
