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
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

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

# 3. CONFIGURE LLM
# temperature=0 makes it factual (good for research)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# 4. DEFINE AGENT STATE (Memory)
class AgentState(TypedDict):
    # Ensure new messages are appended to history
    messages: Annotated[list, add_messages]


# 5. DEFINE NODES

def chatbot(state: AgentState):
    """
    The 'Brain' node. Decides what to do.
    """
    system_instruction = SystemMessage(
        content="""
        You are a helpful research assistant. You have access to a search engine (Tavily).
        
        CRITICAL INSTRUCTIONS:
        - If the user asks for real-time information (like stock prices, weather, news), you MUST use the search tool.
        - Do NOT refuse to answer. Do NOT say "I cannot provide real-time info".
        - Just search, read the results, and provide the answer.
        """
    )
    
    # Add system message to history
    messages = [system_instruction] + state["messages"]
    
    # Invoke the LLM
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


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
            print("-" * 50)
