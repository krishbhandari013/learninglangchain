try:
    from langchain.agents import create_agent
    print("langchain.agents.create_agent exists")
except ImportError:
    print("langchain.agents.create_agent does NOT exist")

try:
    from langchain_tavily import TavilySearch
    print("langchain_tavily.TavilySearch exists")
except ImportError:
    print("langchain_tavily.TavilySearch does NOT exist")

try:
    from langgraph.prebuilt import create_react_agent
    print("langgraph.prebuilt.create_react_agent exists")
except ImportError:
    print("langgraph.prebuilt.create_react_agent does NOT exist")
