import streamlit as st
import os

if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from agent import app, get_last_response_metadata, get_policy_snapshot, submit_feedback

# --- PAGE SETUP ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🧐")
st.title("Deep Research Agent 🧐")
st.markdown("Ask me anything. I will search the web, read pages, and give you a report.")

with st.sidebar:
    st.subheader("Learning Mode")
    st.caption("The assistant adapts its response style using reward feedback after each answer.")
    if st.button("Refresh policy view"):
        st.rerun()

    policy_snapshot = get_policy_snapshot()
    top_actions = {
        state: max(actions, key=actions.get)
        for state, actions in policy_snapshot.items()
        if actions
    }
    st.json(top_actions)

# --- SESSION STATE (Memory) ---
# 1. Create a unique Thread ID for this specific user session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


def _reward_for_feedback(feedback_value: str) -> float:
    return 1.0 if feedback_value == "up" else -1.0


def _render_feedback_controls(message: dict):
    if message["role"] != "assistant":
        return

    message_id = message.get("id", str(abs(hash(message.get("content", "")))))
    feedback_key = f"feedback_{message_id}"
    if message.get("feedback"):
        status = "Helpful" if message["feedback"] == "up" else "Needs work"
        st.caption(f"Feedback recorded: {status}")
        return

    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("👍", key=f"up_{feedback_key}"):
            reward = _reward_for_feedback("up")
            submit_feedback(st.session_state.thread_id, reward)
            message["feedback"] = "up"
            st.rerun()
    with col2:
        if st.button("👎", key=f"down_{feedback_key}"):
            reward = _reward_for_feedback("down")
            submit_feedback(st.session_state.thread_id, reward)
            message["feedback"] = "down"
            st.rerun()
    with col3:
        st.caption("Rate this response to improve future answers.")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_feedback_controls(message)

# --- HANDLE USER INPUT ---
if prompt := st.chat_input("What would you like to research?"):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
        # Create a status container to show "Thinking..."
        with st.status("🧠 Deep Research in progress...", expanded=True) as status:
            st.write("Initializing agent...")
            
            # Prepare config with the unique thread ID
            config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Run the agent stream
            final_response = None
            
            # We stream the events to see the "steps"
            for event in app.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                # Inspect the event to see what node just finished
                for key, value in event.items():
                    if key == "agent":
                        st.write("🤖 Agent is planning next steps...")
                    elif key == "tools":
                        st.write("🔎 Searching external sources...")
            
            # Update status to complete
            status.update(label="Research Complete!", state="complete", expanded=False)

        # 3. Get and Display Final Answer
        # We fetch the final state from the agent's memory
        final_state = app.get_state(config)
        
        if final_state.values and "messages" in final_state.values:
            metadata = get_last_response_metadata(config)
            final_response = metadata["text"]

            # Implicit reward ensures learning occurs after every response.
            submit_feedback(config, reward=0.05)
            
            st.markdown(final_response)
            st.caption(
                f"Adaptive style used: `{metadata['rl_action']}` for state `{metadata['rl_state']}`"
            )
            _id = str(uuid.uuid4())
            
            # Save to history
            st.session_state.messages.append(
                {
                    "id": _id,
                    "role": "assistant",
                    "content": final_response,
                    "feedback": None,
                }
            )