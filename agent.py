# agent.py
# -----------------------------
# Deep Research Agent
# -----------------------------

# 1. IMPORTS
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Annotated, Any, cast
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from rl_policy import InteractionMemory, ReinforcementPolicy

# Load .env for local development
load_dotenv()


# 2. CONFIGURE TOOLS

# Get Tavily API key from environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise ValueError(
        "GOOGLE_API_KEY not found! Set it in your environment, .env file, or Streamlit secrets."
    )

# Initialize TavilySearch tool when a key is available.
search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_key) if tavily_key else None
tools = [search_tool] if search_tool else []

# Initialize lightweight RL policy (persisted across runs)
rl_policy = ReinforcementPolicy(file_path="rl_policy.json")
interaction_memory = InteractionMemory(file_path="interaction_memory.json")

# 3. CONFIGURE LLM
# temperature=0 makes it factual (good for research)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = cast(Any, llm.bind_tools(tools))


# 4. DEFINE AGENT STATE (Memory)
class AgentState(TypedDict):
    # Ensure new messages are appended to history
    messages: Annotated[list[Any], add_messages]
    rl_state: NotRequired[str]
    rl_action: NotRequired[str]


class ResponseMetadata(TypedDict):
    text: str
    rl_state: str
    rl_action: str
    used_search: bool
    response_mode: str


def _current_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


PREDICTION_KEYWORDS = [
    "predict",
    "prediction",
    "who would win",
    "who will win",
    "score",
    "scoreline",
    "fixture",
    "match",
    "tomorrow",
    "next game",
    "next match",
]

TIME_SENSITIVE_KEYWORDS = [
    "latest",
    "recent",
    "today",
    "yesterday",
    "this week",
    "world cup",
    "winner",
    "champion",
    "standings",
    "result",
    "score",
    "match",
    "tournament",
    "news",
    "2026",
    "2025",
    "2024",
]


def _is_prediction_request(text: str) -> bool:
    content = text.lower()
    return any(keyword in content for keyword in PREDICTION_KEYWORDS)


def _is_time_sensitive_request(text: str) -> bool:
    content = text.lower()
    return any(keyword in content for keyword in TIME_SENSITIVE_KEYWORDS)


def _get_latest_human_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _used_search_in_latest_turn(messages: list[Any]) -> bool:
    latest_human_index = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            latest_human_index = index
            break

    if latest_human_index is None:
        return False

    for msg in messages[latest_human_index + 1:]:
        if isinstance(msg, ToolMessage):
            return True
    return False


def _response_mode_for_turn(messages: list[Any]) -> tuple[bool, str]:
    latest_user_text = _get_latest_human_text(messages)
    used_search = _used_search_in_latest_turn(messages)
    is_prediction = _is_prediction_request(latest_user_text)

    if used_search and is_prediction:
        return True, "live_search_prediction"
    if used_search:
        return True, "live_search"
    if is_prediction:
        return False, "model_prediction"
    return False, "model_only"


def _build_system_instruction() -> SystemMessage:
    current_date = _current_utc_date()
    search_guidance = (
        "You have access to a search engine (Tavily). You must use it for real-time or time-sensitive questions, including recent news, "
        "today, yesterday, this week, live events, match results, standings, transfer news, injuries, and what happened on a specific date. "
        "For recent news, prioritize the newest sources and do not answer from stale model memory. "
        "For future sports matches or fixtures, search for recent form, injuries, squad news, and other current context before answering."
        if search_tool
        else "You do not currently have a live search tool configured, so answer from model knowledge and be explicit when information may be outdated."
    )
    return SystemMessage(
        content=(
            "You are a helpful research assistant. "
            f"The current UTC date is {current_date}. "
            "Interpret absolute dates relative to that date. "
            "If a user asks about a date earlier than the current date, treat it as the past, not the future. "
            "If a user asks about a date later than the current date, treat it as future-looking. "
            "Do not claim a past date is in the future. "
            "If the user asks for recent news, latest updates, or a recent dated event, do not rely on old background knowledge. "
            "Use search and summarize the freshest relevant reports. "
            "If the user asks you to predict a future match, scoreline, or winner, provide a prediction rather than a fact. "
            "State the likely winner, include an estimated score only if asked or useful, and clearly label it as a prediction with uncertainty. "
            "Base sports predictions on current evidence when search is available, not on outdated memory alone. "
            f"{search_guidance}"
        )
    )


# 5. DEFINE NODES

def chatbot(state: AgentState) -> dict[str, Any]:
    """
    The 'Brain' node. Decides what to do.
    """
    latest_user_message = _get_latest_human_text(state["messages"])

    selected_state = rl_policy.classify_state(latest_user_message)
    selected_action = rl_policy.choose_action(selected_state)
    style_instruction = rl_policy.get_style_instruction(selected_action)

    system_instruction = _build_system_instruction()

    strategy_instruction = SystemMessage(
        content=(
            "Adaptive response policy selected for this turn: "
            f"'{selected_action}'. {style_instruction}"
        )
    )

    freshness_instruction: list[SystemMessage] = []
    if search_tool and _is_time_sensitive_request(latest_user_message):
        try:
            # Hard freshness guard: fetch current web context for time-sensitive prompts.
            fresh_context = search_tool.invoke(latest_user_message)
            freshness_instruction = [
                SystemMessage(
                    content=(
                        "Fresh web context (Tavily) for this query is provided below. "
                        "Use it to ground claims about dates, winners, and recent news. "
                        f"Context: {fresh_context}"
                    )
                )
            ]
        except Exception:
            # If search fails transiently, continue with normal flow.
            freshness_instruction = []
    
    # Add system message to history
    messages: list[Any] = [system_instruction, strategy_instruction] + freshness_instruction + state["messages"]
    
    # Invoke the LLM
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "rl_state": selected_state,
        "rl_action": selected_action,
    }


def _extract_text(content: Any) -> str:
    """Normalize provider-specific message content formats to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                chunk = item.get("text")
                if chunk:
                    text_chunks.append(str(chunk))
            elif item:
                text_chunks.append(str(item))
        return "\n".join(text_chunks)
    return str(content)


def _thread_id_from_config(config: RunnableConfig | str) -> str:
    if isinstance(config, str):
        return config
    configurable = config.get("configurable", {})
    if configurable.get("thread_id"):
        return str(configurable["thread_id"])
    return "1"


def _config_for_thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def get_last_response_metadata(config: RunnableConfig | str) -> ResponseMetadata:
    """Return latest assistant text and RL metadata for a thread."""
    thread_id = _thread_id_from_config(config)
    snapshot = app.get_state(_config_for_thread(thread_id))
    if not snapshot.values:
        return {
            "text": "",
            "rl_state": "general",
            "rl_action": "detailed",
            "used_search": False,
            "response_mode": "model_only",
        }

    values = snapshot.values
    messages = cast(list[Any], values.get("messages", []))
    last_content = messages[-1].content if messages else ""
    used_search, response_mode = _response_mode_for_turn(messages)
    return {
        "text": _extract_text(last_content),
        "rl_state": str(values.get("rl_state", "general")),
        "rl_action": str(values.get("rl_action", "detailed")),
        "used_search": used_search,
        "response_mode": response_mode,
    }


def submit_feedback(config: RunnableConfig | str, reward: float) -> float:
    """Apply reward to the latest policy choice for the thread."""
    metadata = get_last_response_metadata(config)
    return rl_policy.update(
        state=metadata["rl_state"],
        action=metadata["rl_action"],
        reward=reward,
        next_state=metadata["rl_state"],
    )


def get_policy_snapshot() -> dict[str, dict[str, float]]:
    return rl_policy.snapshot()


def record_interaction(config: RunnableConfig | str, user_message: str):
    """Create a structured learning record for the latest assistant turn."""
    metadata = get_last_response_metadata(config)
    return interaction_memory.start_interaction(
        thread_id=_thread_id_from_config(config),
        user_message=user_message,
        assistant_response=metadata["text"],
        rl_state=metadata["rl_state"],
        rl_action=metadata["rl_action"],
    )


def learn_from_feedback(
    config: RunnableConfig | str,
    feedback_text: str | None = None,
    reward: float | None = None,
    entry_id: str | None = None,
):
    """Update the latest recorded interaction from explicit feedback or user reaction text."""
    thread_id = _thread_id_from_config(config)
    entry = interaction_memory.apply_feedback(
        feedback_text=feedback_text,
        explicit_reward=reward,
        entry_id=entry_id,
        thread_id=thread_id,
    )
    if not entry or entry["reward"] == 0:
        return entry

    rl_policy.update(
        state=entry["rl_state"],
        action=entry["rl_action"],
        reward=entry["reward"],
        next_state=entry["rl_state"],
    )
    return entry


def learn_from_followup(config: RunnableConfig | str, user_message: str):
    """Treat clear satisfaction or dissatisfaction in the next user message as feedback."""
    return learn_from_feedback(config, feedback_text=user_message)


def get_interaction_memory_snapshot() -> dict[str, Any]:
    return interaction_memory.snapshot()


# ToolNode automatically executes whatever tools the LLM asks for
tool_node = ToolNode(tools)


# 6. BUILD WORKFLOW
workflow = StateGraph(AgentState)

# Add main nodes
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)

# Entry point
workflow.add_edge(START, "agent")

# Conditional edge: use tools if LLM asks
workflow.add_conditional_edges("agent", tools_condition)

# Loop back from tools to agent
workflow.add_edge("tools", "agent")

# Compile workflow with memory saver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Optional: generate graph image for visualization
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(png_data)
    print("📸 Graph saved as 'agent_graph.png'")
except Exception as e:
    print(f"Could not draw graph: {e}")


# 7. MAIN (for CLI testing)
if __name__ == "__main__":
    print("🚀 Deep Research Agent is ON (with Memory).")
    print("Type 'quit' or 'exit' to stop.\n")
    
    thread: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        learn_from_followup(thread, user_input)
        inputs: AgentState = {"messages": [HumanMessage(content=user_input)]}
        
        # Stream the output
        for event in app.stream(inputs, config=thread):
            for key in event.keys():
                print(f"  --> Node '{key}' finished.")
        
        # Fetch final answer
        snapshot = app.get_state(thread)
        if snapshot.values and "messages" in snapshot.values:
            last_msg = snapshot.values["messages"][-1]
            content = last_msg.content
            
            print("\n🤖 Final Answer:")
            if isinstance(content, list):
                print(_extract_text(content))
            else:
                print(content)

            entry = record_interaction(thread, user_input)

            user_feedback = input("Feedback (+ helpful / - not helpful / Enter skip): ").strip()
            if user_feedback == "+":
                learn_from_feedback(thread, reward=1.0, entry_id=entry["id"])
            elif user_feedback == "-":
                learn_from_feedback(thread, reward=-1.0, entry_id=entry["id"])

            print("-" * 50)
