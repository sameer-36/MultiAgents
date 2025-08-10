# multiagents.py
import os
import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.googlesearch import GoogleSearchTools

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing from environment variables.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Agent Definitions ---
@st.cache_resource
def get_agents():
    """Initialize and return all agents."""
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=Groq(id="qwen/qwen3-32b"),
        tools=[DuckDuckGoTools()],
        instructions="Always include the sources.",
        show_tool_calls=True,
        markdown=True,
    )

    google_agent = Agent(
        name="Google News Agent",
        tools=[GoogleSearchTools()],
        description="Find the latest news about a given topic.",
        model=Groq(id="qwen/qwen3-32b"),
        instructions=[
            "Given a topic, respond with 4 latest news items.",
            "Search for 10 news items and select the top 4 unique ones.",
            "Search in English and French."
        ],
        show_tool_calls=True,
        debug_mode=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=Groq(id="qwen/qwen3-32b"),
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True
        )],
        instructions="Use tables to display data.",
        show_tool_calls=True,
        markdown=True,
    )

    agent_team = Agent(
        team=[google_agent, finance_agent],
        model=Groq(id="qwen/qwen3-32b"),
        instructions=["Always include sources.", "Use tables to display data."],
        show_tool_calls=True,
        markdown=True,
    )

    return web_agent, google_agent, finance_agent, agent_team


web_agent, google_agent, finance_agent, agent_team = get_agents()

# --- Streamlit UI ---
st.set_page_config(page_title="Finance & Web Agent", page_icon="💹")

st.title("💹 Multi-Agent Financial & Web Analysis Tool")
st.write("Ask me about companies, markets, or any other financial info.")

st.sidebar.header("Agent Selection")
agent_choice = st.sidebar.radio(
    "Choose which agent to use:",
    ("Web Agent", "Finance Agent", "Both (Team)")
)

if agent_choice == "Web Agent":
    selected_agent = google_agent
elif agent_choice == "Finance Agent":
    selected_agent = finance_agent
else:
    selected_agent = agent_team

user_query = st.text_area(
    "Enter your query:",
    placeholder="Example: Analyze Tesla, NVDA, and Apple for long-term investment"
)

if st.button("Run Analysis"):
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner(f"Running {agent_choice}... please wait"):
            try:
                result = selected_agent.run(user_query)
                if result and hasattr(result, "content"):
                    st.markdown(result.content)
                else:
                    st.warning("No content returned from the agent.")
            except Exception as e:
                st.error(f"Error: {e}")
