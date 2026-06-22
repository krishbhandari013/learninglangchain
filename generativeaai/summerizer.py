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
# TOOL
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
# AGENT (NEW API)
# -------------------
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

# -------------------
# THREAD
# -------------------
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

# -------------------
# SIMPLE SUMMARIZER
# -------------------
def summarize_if_needed(messages, llm):
    if len(messages) < 12:
        return messages

    prompt = [
        ("system", "Summarize this conversation concisely, keep key facts."),
        *messages
    ]

    summary = llm.invoke(prompt).content

    return [("system", f"Conversation summary: {summary}")]

# -------------------
# CHAT LOOP
# -------------------
while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # ✅ Get previous messages
    state = memory.get(config)
    history = state["messages"] if state else []

    # ✅ Apply summarization middleware manually
    summarized_history = summarize_if_needed(history, model)

    # ✅ Call agent with summarized context
    result = agent.invoke(
        {
            "messages": summarized_history + [("user", user_input)]
        },
        config=config
    )

    print("\nAI:", result["messages"][-1].content)
