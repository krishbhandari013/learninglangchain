from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver

# -------------------
# MODEL
# -------------------
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# -------------------
# TOOL (web search)
# -------------------
search = TavilySearch(max_results=3)

@tool
def web_search(query: str):
    """Search the web for latest information"""
    return search.invoke(query)

tools = [web_search]

# -------------------
# MEMORY
# -------------------
memory = InMemorySaver()

# -------------------
# AGENT
# -------------------
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

# -------------------
# THREAD (memory id)
# -------------------
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

# -------------------
# CHAT LOOP
# -------------------
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                ("user", user_input)
            ]
        },
        config=config
    )

    print("\nAI:", result["messages"][-1].content)
