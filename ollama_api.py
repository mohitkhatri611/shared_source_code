# Step 1: Define tools and model
from langchain.tools import tool
from langchain_ollama import ChatOllama
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Ollama local model (pick one you have; Ministral is great for agents)
model = ChatOllama(model="ministral-3:8b", temperature=0)


# ---- Tools ----
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` by `b`."""
    return a / b


@tool
def get_current_time(tz: str = "Asia/Kolkata") -> str:
    """Get the current time in ISO format for a timezone (default: Asia/Kolkata)."""
    now = datetime.now(ZoneInfo(tz))
    return now.isoformat(timespec="seconds")


@tool
def search(query: str, top_k: int = 3) -> str:
    """
    Return SAMPLE search results (stub) as JSON.
    Replace this with a real search API later if you want live results.
    """
    # Sample results (hardcoded) — good for demo/testing agent loops
    sample = [
        {
            "title": "LangGraph Quickstart (Docs by LangChain)",
            "url": "https://docs.langchain.com/oss/python/langgraph/quickstart",
            "snippet": "Build a calculator agent using the LangGraph Graph API with tools + a loop.",
        },
        {
            "title": "Ollama Tool Calling",
            "url": "https://docs.ollama.com/capabilities/tool-calling",
            "snippet": "Ollama supports tool calling (function calling) and multi-turn agent loops.",
        },
        {
            "title": "LangChain Ollama Integration Reference",
            "url": "https://reference.langchain.com/python/integrations/langchain_ollama/",
            "snippet": "ChatOllama supports binding tools for tool-calling compatible Ollama models.",
        },
    ]
    return json.dumps(
        {"query": query, "top_k": top_k, "results": sample[:top_k]}, indent=2
    )


# Augment the LLM with tools (same idea as quickstart)
tools = [add, multiply, divide, get_current_time, search]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# Step 3: Define model node
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a helpful assistant.\n"
                            "- Use tools when needed.\n"
                            "- For math, use add/multiply/divide.\n"
                            "- For time, use get_current_time.\n"
                            "- For searching, use search.\n"
                            "If you call a tool, wait for the tool result before answering."
                        )
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Step 4: Define tool node
from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    last = state["messages"][-1]

    for tool_call in last.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        observation = tool_fn.invoke(tool_call["args"])

        # ToolMessage content should be a string
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


# Step 5: Define logic to determine whether to end
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Route to tool node if LLM made a tool call; else end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# Step 6: Build and compile the agent
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Invoke (demo)
from langchain.messages import HumanMessage

messages = [
    HumanMessage(
        content=(
            "1) What time is it right now in India?\n"
            "2) Search: 'LangGraph Quickstart full code example' and summarize in 1 line.\n"
            "3) Also compute: (11434 + 12341) * 2."
        )
    )
]

out = agent.invoke({"messages": messages})

print("\n" + "=" * 40)
print("  Conversation Transcript")
print("=" * 40)

for m in out["messages"]:
    # Check string representation or type to identify message role
    msg_type = m.type
    content = m.content

    if msg_type == "human":
        print(f"\n👤 USER:\n{content}")
    elif msg_type == "ai":
        if m.tool_calls:
            for tc in m.tool_calls:
                print(
                    f"\n🤖 AGENT (Thinking):\nCalling tool '{tc['name']}' with {tc['args']}"
                )
        else:
            print("\n" + "=" * 70)
            print("\n" + "=" * 70)
            print(f"\n🤖 AGENT:\n{content}")
    elif msg_type == "tool":
        print(f"\n🛠️  TOOL OUTPUT:\n{content}")

print("\n" + "=" * 70)
print(f"Total LLM calls: {out.get('llm_calls')}")
