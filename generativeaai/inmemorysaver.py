from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated


# -----------------------
# MODEL
# -----------------------
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# -----------------------
# TAVILY TOOL
# -----------------------
search = TavilySearch(max_results=3)


@tool
def web_search(query: str):
    """Search latest information from web"""
    return search.invoke(query)


tools = [web_search]

model_with_tools = model.bind_tools(tools)


# -----------------------
# STATE
# -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# -----------------------
# CHAT NODE
# -----------------------
def chatbot(state: State):
    return {
        "messages": [model_with_tools.invoke(state["messages"])]
    }


# -----------------------
# GRAPH
# -----------------------
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

builder.add_edge("tools", "chatbot")


# -----------------------
# MEMORY
# -----------------------
memory = InMemorySaver()

graph = builder.compile(checkpointer=memory)


# -----------------------
# CHAT LOOP
# -----------------------
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = graph.invoke(
        {
            "messages": [
                ("user", user_input)
            ]
        },
        config=config
    )

    print("\nAI:", result["messages"][-1].content)