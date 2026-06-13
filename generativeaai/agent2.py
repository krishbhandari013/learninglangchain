import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

# ==================================================
# LLM
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# ==================================================
# TOOLS
# ==================================================

# Search Tool
tavily_search = TavilySearch()

# Arithmetic Tools
@tool
def add(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtracts b from a."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divides a by b."""
    if b == 0:
        return "Error: Division by zero."
    return a / b

tools = [tavily_search, add, subtract, multiply, divide]

# ==================================================
# CREATE AGENT
# ==================================================

agent = create_agent(
    model=llm,
    tools=tools
)

# ==================================================
# CHAT LOOP
# ==================================================


while True:
    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    final_response = None

    for response in agent.stream({"messages": query},stream_mode="values"):
        final_response = response

    print("\nDEBUG:")
    print(final_response["messages"][-1].content)
