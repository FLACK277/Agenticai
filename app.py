import streamlit as st
import os
from typing import Any, NotRequired, TypedDict, cast

def _load_secret_from_streamlit(name: str) -> None:
    if name not in st.secrets:
        return

    value = str(st.secrets[name]).strip()
    if not value or value.startswith("your_"):
        return

    os.environ.setdefault(name, value)


_load_secret_from_streamlit("TAVILY_API_KEY")
_load_secret_from_streamlit("GOOGLE_API_KEY")

import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from agent import (
    app,
    get_interaction_memory_snapshot,
    get_last_response_metadata,
    get_policy_snapshot,
    learn_from_feedback,
    learn_from_followup,
    record_interaction,
)


class ChatMessage(TypedDict):
    role: str
    content: str
    id: NotRequired[str]
    feedback: NotRequired[str | None]
    feedback_note: NotRequired[str]
    learning_entry_id: NotRequired[str]
    response_mode: NotRequired[str]
    used_search: NotRequired[bool]

# --- PAGE SETUP ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🧐")
st.title("Deep Research Agent 🧐")
st.markdown("Ask me anything. I will search the web, read pages, and give you a report.")

with st.sidebar:
    st.subheader("Learning Mode")
    st.caption("The assistant stores a structured lesson after every answer and updates it from user reactions.")
    if st.button("Refresh policy view"):
        st.rerun()

    policy_snapshot = get_policy_snapshot()
    top_actions = {
        state: max(actions.items(), key=lambda item: item[1])[0]
        for state, actions in policy_snapshot.items()
        if actions
    }
    st.json(top_actions)

    memory_snapshot = get_interaction_memory_snapshot()
    st.caption(f"Stored learning records: {len(memory_snapshot['entries'])}")

# --- SESSION STATE (Memory) ---
# 1. Create a unique Thread ID for this specific user session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

messages = cast(list[ChatMessage], st.session_state["messages"])


def _reward_for_feedback(feedback_value: str) -> float:
    return 1.0 if feedback_value == "up" else -1.0


def _response_mode_label(message: ChatMessage) -> str | None:
    mode = message.get("response_mode")
    if mode == "live_search_prediction":
        return "Mode: live search + prediction"
    if mode == "live_search":
        return "Mode: live search"
    if mode == "model_prediction":
        return "Mode: model prediction"
    if mode == "model_only":
        return "Mode: model only"
    return None


def _render_feedback_controls(message: ChatMessage):
    if message["role"] != "assistant":
        return

    mode_label = _response_mode_label(message)
    if mode_label:
        st.caption(mode_label)

    message_id = message.get("id", str(abs(hash(message.get("content", "")))))
    feedback_key = f"feedback_{message_id}"
    feedback_value = message.get("feedback")
    if feedback_value:
        status = "Helpful" if feedback_value == "up" else "Needs work"
        st.caption(f"Feedback recorded: {status}")
        prompt_text = (
            "What was useful about this answer?"
            if feedback_value == "up"
            else "What should the agent improve next time?"
        )
        default_note = message.get("feedback_note", "")
        note = st.text_input(prompt_text, value=default_note, key=f"note_{feedback_key}")
        if st.button("Save feedback detail", key=f"save_note_{feedback_key}"):
            reward = _reward_for_feedback(feedback_value)
            detail = note.strip()
            feedback_text = (
                f"Positive feedback: {detail}" if feedback_value == "up" else f"Negative feedback: {detail}"
            )
            learn_from_feedback(
                st.session_state.thread_id,
                reward=reward,
                feedback_text=feedback_text,
                entry_id=message.get("learning_entry_id"),
            )
            message["feedback_note"] = detail
            st.rerun()
        if default_note:
            st.caption(f"Saved detail: {default_note}")
        return

    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("👍", key=f"up_{feedback_key}"):
            reward = _reward_for_feedback("up")
            learn_from_feedback(
                st.session_state.thread_id,
                reward=reward,
                feedback_text="Positive feedback",
                entry_id=message.get("learning_entry_id"),
            )
            message["feedback"] = "up"
            message["feedback_note"] = ""
            st.rerun()
    with col2:
        if st.button("👎", key=f"down_{feedback_key}"):
            reward = _reward_for_feedback("down")
            learn_from_feedback(
                st.session_state.thread_id,
                reward=reward,
                feedback_text="Negative feedback",
                entry_id=message.get("learning_entry_id"),
            )
            message["feedback"] = "down"
            message["feedback_note"] = ""
            st.rerun()
    with col3:
        st.caption("Rate this response to improve future answers.")

# --- DISPLAY CHAT HISTORY ---
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_feedback_controls(message)

# --- HANDLE USER INPUT ---
if prompt := st.chat_input("What would you like to research?"):
    learn_from_followup(st.session_state.thread_id, prompt)

    # 1. Display User Message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
        # Create a status container to show "Thinking..."
        with st.status("🧠 Deep Research in progress...", expanded=True) as status:
            st.write("Initializing agent...")
            
            # Prepare config with the unique thread ID
            config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # We stream the events to see the "steps"
            for event in cast(Any, app).stream({"messages": [HumanMessage(content=prompt)]}, config=config):
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
            final_response: str = metadata["text"]
            interaction_entry = record_interaction(config, prompt)
            
            st.markdown(final_response)
            st.caption(
                f"Adaptive style used: `{metadata['rl_action']}` for state `{metadata['rl_state']}`"
            )
            _id = str(uuid.uuid4())
            
            # Save to history
            messages.append(
                {
                    "id": _id,
                    "role": "assistant",
                    "content": final_response,
                    "learning_entry_id": interaction_entry["id"],
                    "feedback": None,
                    "feedback_note": "",
                    "response_mode": metadata["response_mode"],
                    "used_search": metadata["used_search"],
                }
            )