# sustainability_taskforce.py
import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from datetime import datetime

from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.hackernews import HackerNewsTools

# --- Load environment variables ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# --- Custom Tool: CSV Data Analysis ---
class CSVAnalysisTool:
    def run(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            summary = df.describe().to_markdown()
            return f"📊 Data Analysis Summary:\n{summary}", df
        except Exception as e:
            return f"❌ Error analyzing dataset: {e}", None

# --- Define Agents ---
news_agent = Agent(
    name="News Analyst 📰",
    role="Find recent sustainability news",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Search for city-level green projects in the past year.",
    show_tool_calls=True,
    markdown=True,
)

data_agent = Agent(
    name="Data Analyst 📊",
    role="Analyze environmental datasets",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[CSVAnalysisTool()],
    instructions="Read and summarize air quality/environmental CSV data trends.",
    show_tool_calls=True,
    markdown=True,
)

policy_agent = Agent(
    name="Policy Reviewer 🏛️",
    role="Summarize government sustainability policies",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[GoogleSearchTools()],
    instructions="Search official government sites for city sustainability policy updates.",
    show_tool_calls=True,
    markdown=True,
)

innovation_agent = Agent(
    name="Innovations Scout 💡",
    role="Find innovative green tech ideas",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[HackerNewsTools(), GoogleSearchTools()],
    instructions="Search for innovative urban sustainability technologies.",
    show_tool_calls=True,
    markdown=True,
)

# --- Team Agent ---
taskforce_team = Team(
    name="🌱 Sustainability Task Force",
    mode="collaborate",
    model=Groq(id="qwen/qwen3-32b"),
    members=[news_agent, data_agent, policy_agent, innovation_agent],
    instructions=["Work together to propose a sustainability plan for the city."],
    show_tool_calls=True,
    markdown=True,
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sustainability Task Force", page_icon="🌱", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
        .stApp {background-color: #f5f7fa;}
        .main-title {text-align:center; font-size:40px; color:#2e7d32; font-weight:bold;}
        .subtitle {text-align:center; font-size:18px; color:#555;}
        .card {background:white; padding:20px; border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.1); margin-bottom:20px;}
        .agent-header {font-size:22px; font-weight:bold; color:#1b5e20;}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">🌍 Sustainability Task Force</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered multi-agent system for smarter, greener cities</div>', unsafe_allow_html=True)
st.markdown("")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
agent_choice = st.sidebar.selectbox(
    "Select Agent Mode:",
    ("News Analyst 📰", "Data Analyst 📊", "Policy Reviewer 🏛️", "Innovations Scout 💡", "🌱 All Agents (Task Force)")
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Choose **Task Force** for a full sustainability proposal.")

# --- Input Section ---
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✍️ Research Topic")
    topic = st.text_area("Enter your sustainability research topic:",
                         placeholder="Example: How can our city improve air quality?",
                         height=120)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Data Upload Section ---
uploaded_file = None
if agent_choice == "Data Analyst 📊":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Upload Environmental Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file).head()
        st.markdown("🔍 Preview of uploaded data:")
        st.dataframe(df_preview)
    st.markdown('</div>', unsafe_allow_html=True)

# --- History ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Run Button ---
if st.button("🚀 Run Analysis", use_container_width=True):
    if topic.strip() or (agent_choice == "Data Analyst 📊" and uploaded_file):
        with st.spinner(f"Running {agent_choice}... please wait ⏳"):
            try:
                if agent_choice == "Data Analyst 📊" and uploaded_file:
                    file_path = Path("tmp_uploaded.csv")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    result, df = data_agent.tools[0].run(str(file_path))

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Data Analyst Report")
                    st.markdown(result)

                    # Show chart
                    if df is not None:
                        st.markdown("#### 📈 Trend Chart")
                        fig, ax = plt.subplots()
                        df.select_dtypes(include="number").plot(ax=ax)
                        st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    selected_agent = {
                        "News Analyst 📰": news_agent,
                        "Data Analyst 📊": data_agent,
                        "Policy Reviewer 🏛️": policy_agent,
                        "Innovations Scout 💡": innovation_agent,
                        "🌱 All Agents (Task Force)": taskforce_team,
                    }[agent_choice]

                    result = selected_agent.run(topic)
                    if result and hasattr(result, "content"):
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"<div class='agent-header'>{agent_choice} Results</div>", unsafe_allow_html=True)
                        if agent_choice == "🌱 All Agents (Task Force)":
                            st.success("✅ Full Sustainability Proposal Generated")
                            st.markdown("### 📑 Sustainability Proposal")
                        st.markdown(result.content)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.session_state.history.append(
                            {"time": datetime.now().strftime("%H:%M:%S"),
                             "agent": agent_choice,
                             "topic": topic,
                             "result": result.content}
                        )
                    else:
                        st.warning("⚠️ No content returned from the agent.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a topic or upload a dataset.")

# --- Past Results ---
if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📜 Past Results")
    for entry in reversed(st.session_state.history):
        with st.expander(f"{entry['time']} - {entry['agent']} on '{entry['topic']}'"):
            st.markdown(entry["result"])
    st.markdown('</div>', unsafe_allow_html=True)

# --- Export Section ---
if st.session_state.history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⬇️ Export Last Result")
    last_result = st.session_state.history[-1]["result"]

    st.download_button(
        "📄 Download as Markdown",
        data=last_result,
        file_name="sustainability_report.md"
    )
    st.download_button(
        "📝 Download as Text",
        data=last_result,
        file_name="sustainability_report.txt"
    )
    st.markdown('</div>', unsafe_allow_html=True)
