from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import asyncio

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)

# Configure MCP servers
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": r"C:\Python311\python.exe",
            "args": [
                r"C:\Users\dell\OneDrive\Documents\Desktop\mcp-math-local-server\main.py"
            ],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://expense-tracker-mcp-23.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('EXPENSE_TRACKER_KEY')}"
            }
        }
    }
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()

    print("\nLoaded MCP Tools:")
    for tool in tools:
        print(tool.name)

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile()

async def main():

    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Give me all my expenses for the month of Nov from 1 Nov to 30 Nov"
                )
            ]
        }
    )

    print("\nFinal Response:")
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())