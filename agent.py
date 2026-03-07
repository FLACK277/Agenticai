# agent.py
# -----------------------------
# Deep Research Agent
# -----------------------------

# 1. IMPORTS
import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from rl_policy import ReinforcementPolicy

# Load .env for local development
load_dotenv()


# 2. CONFIGURE TOOLS

# Get Tavily API key from environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError(
        "TAVILY_API_KEY not found! Set it in your environment or Streamlit secrets."
    )

# Initialize TavilySearch tool
search_tool = TavilySearch(max_results=3, tavily_api_key=tavily_key)
tools = [search_tool]

# Initialize lightweight RL policy (persisted across runs)
rl_policy = ReinforcementPolicy(file_path="rl_policy.json")

# 3. CONFIGURE LLM
# temperature=0 makes it factual (good for research)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# 4. DEFINE AGENT STATE (Memory)
class AgentState(TypedDict):
    # Ensure new messages are appended to history
    messages: Annotated[list, add_messages]
    rl_state: NotRequired[str]
    rl_action: NotRequired[str]


# 5. DEFINE NODES

def chatbot(state: AgentState):
    """
    The 'Brain' node. Decides what to do.
    """
    latest_user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    selected_state = rl_policy.classify_state(latest_user_message)
    selected_action = rl_policy.choose_action(selected_state)
    style_instruction = rl_policy.get_style_instruction(selected_action)

    system_instruction = SystemMessage(
        content="""
        You are a helpful research assistant. You have access to a search engine (Tavily).
        
        CRITICAL INSTRUCTIONS:
        - If the user asks for real-time information (like stock prices, weather, news), you MUST use the search tool.
        - Do NOT refuse to answer. Do NOT say "I cannot provide real-time info".
        - Just search, read the results, and provide the answer.
        """
    )

    strategy_instruction = SystemMessage(
        content=(
            "Adaptive response policy selected for this turn: "
            f"'{selected_action}'. {style_instruction}"
        )
    )
    
    # Add system message to history
    messages = [system_instruction, strategy_instruction] + state["messages"]
    
    # Invoke the LLM
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "rl_state": selected_state,
        "rl_action": selected_action,
    }


def _extract_text(content):
    """Normalize provider-specific message content formats to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
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
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
        if isinstance(configurable, dict) and configurable.get("thread_id"):
            return str(configurable["thread_id"])
    return "1"


def _config_for_thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def get_last_response_metadata(config: RunnableConfig | str):
    """Return latest assistant text and RL metadata for a thread."""
    thread_id = _thread_id_from_config(config)
    snapshot = app.get_state(_config_for_thread(thread_id))
    if not snapshot.values:
        return {
            "text": "",
            "rl_state": "general",
            "rl_action": "detailed",
        }

    values = snapshot.values
    messages = values.get("messages", [])
    last_content = messages[-1].content if messages else ""
    return {
        "text": _extract_text(last_content),
        "rl_state": values.get("rl_state", "general"),
        "rl_action": values.get("rl_action", "detailed"),
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


def get_policy_snapshot():
    return rl_policy.snapshot()


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
                print(content[0]['text'])
            else:
                print(content)

            # Always learn a little after each answer.
            submit_feedback(thread, reward=0.05)

            user_feedback = input("Feedback (+ helpful / - not helpful / Enter skip): ").strip()
            if user_feedback == "+":
                submit_feedback(thread, reward=1.0)
            elif user_feedback == "-":
                submit_feedback(thread, reward=-1.0)

            print("-" * 50)
