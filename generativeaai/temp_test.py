import os

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool, tool
from langchain_community.utilities import SerpAPIWrapper

from dotenv import load_dotenv

# ==================================================
# API KEYS
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
# GOOGLE SEARCH TOOL
# ==================================================

search = SerpAPIWrapper()

@tool
def google_search(query: str) -> str:
    """Useful for searching current information from Google."""
    return search.run(query)

# ==================================================
# CUSTOM TOOL
# ==================================================

@tool
def get_product_price(product_name: str):
    """Get the price of a product."""

    products = {
        "iPhone 17": "$1200",
        "Samsung S30": "$1000",
        "MacBook Pro": "$2000"
    }

    return products.get(product_name, "Product not found")

# ==================================================
# CREATE AGENT
# ==================================================

agent = create_agent(
    model=llm,
    tools=[
        google_search,
        get_product_price
    ]
)

# ==================================================
# CHAT LOOP
# ==================================================

while True:
    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    response = agent.invoke({"messages": [("user", query)]})

    print("\nAI:")
    print(response["messages"][-1].content)