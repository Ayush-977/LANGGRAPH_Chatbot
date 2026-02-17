import os
import uuid
import streamlit as st

from langgraph_backend import chatbot, clear_database
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq

from session_db import init_db, save_session_title, get_all_sessions, delete_all_sessions

# ----------------
# Initialization
# ----------------
st.set_page_config(layout="wide")  # ✅ must be first Streamlit command
init_db()

# -------------
# Helpers
# -------------

def generate_title(first_message_content: str) -> str:
    """Generate a very short title (≤ 4 words, ≤ 30 chars)."""
    try:
        # Ensure your environment has GROQ_API_KEY
        # e.g., export GROQ_API_KEY="your_key_here"
        llm = ChatGroq(model="llama-3.1-8b-instant")
        messages = [
            SystemMessage(content="Return ONLY a concise chat title, MAX 4 words, no quotes."),
            HumanMessage(content=first_message_content),
        ]
        response = llm.invoke(messages)
        title = (response.content or "").strip()
        words = title.split()
        if len(words) > 4:
            title = " ".join(words[:4])
        if len(title) > 30:
            title = title[:27] + "..."
        return title or "New Conversation"
    except Exception:
        return "New Conversation"

def load_convo(thread_id: str):
    """Load message history from LangGraph and convert to Streamlit UI messages."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config) or {}
    except Exception:
        return []
    values = getattr(state, "values", {}) or {}
    messages = values.get("messages", []) or []

    ui_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            ui_msgs.append({"role": "assistant", "content": msg.content})
        # Skip ToolMessage & others in UI history
    return ui_msgs

# -------------------------------
# Session State Initialization
# -------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Provide a default to avoid NameError if sidebar fails
thread_titles = {st.session_state["thread_id"]: "New Chat"}

# -----------
# Sidebar
# -----------
with st.sidebar:
    st.title("LangGraph Chat")

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        save_session_title(new_id, "New Chat")
        st.session_state["thread_id"] = new_id
        st.session_state["message_history"] = []
        st.rerun()

    st.markdown("---")
    st.header("History")

    # Load sessions
    try:
        db_sessions = get_all_sessions()
    except Exception:
        db_sessions = []

    db_sessions = db_sessions[::-1]
    thread_ids = [s[0] for s in db_sessions]
    thread_titles = {s[0]: s[1] for s in db_sessions}

    if st.session_state["thread_id"] not in thread_ids:
        thread_ids.insert(0, st.session_state["thread_id"])
        thread_titles[st.session_state["thread_id"]] = "New Chat"

    try:
        curr_idx = thread_ids.index(st.session_state["thread_id"])
    except ValueError:
        curr_idx = 0

    selected_id = st.radio(
        "Select Chat",
        options=thread_ids,
        format_func=lambda x: thread_titles.get(x, "New Chat"),
        index=curr_idx,
        label_visibility="collapsed",
    )

    if selected_id != st.session_state["thread_id"]:
        st.session_state["thread_id"] = selected_id
        st.session_state["message_history"] = load_convo(selected_id)
        st.rerun()

    # ✅ Proper CSS (no HTML entities)
    st.markdown(
        """
        <style>
        div.stButton > button:first-child { width: 100%; }
        div.stButton > button:last-child { background-color: #ff4b4b; color: white; border: none; }
        div.stButton > button:last-child:hover { background-color: #ff0000; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("⚠️ Clear History"):
        try:
            clear_database()
        finally:
            delete_all_sessions()
            st.session_state.clear()
            st.rerun()

# -------------------
# Main Chat Area
# -------------------

# Show existing messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # Generate & save title for first message or when title is generic
    current_title = thread_titles.get(st.session_state["thread_id"], "New Chat")
    if len(st.session_state["message_history"]) == 0 or current_title == "New Chat":
        new_name = generate_title(user_input)
        save_session_title(st.session_state["thread_id"], new_name)

    # Show the user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Single streaming block with tool status ---
    with st.chat_message("assistant"):
        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        # Fallback for Streamlit versions without st.status
        status_box = None
        try:
            status_box = st.status("Ready", expanded=False)
        except Exception:
            status_placeholder = st.empty()
            def set_status(label, state="running"):
                icon = "🔧" if state == "running" else "✅"
                status_placeholder.info(f"{icon} {label}")

        def unified_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Update tool status if any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or getattr(message_chunk, "tool_name", None) or "tool"
                    if status_box is not None:
                        status_box.update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)
                    else:
                        set_status(f"Using `{tool_name}` …", state="running")

                # Yield only assistant text
                if isinstance(message_chunk, AIMessage):
                    if isinstance(message_chunk.content, str) and message_chunk.content:
                        yield message_chunk.content

        ai_msg = st.write_stream(unified_stream())

        # Finalize status
        try:
            if status_box is not None:
                status_box.update(label="✅ Tool finished", state="complete", expanded=False)
            else:
                set_status("Tool finished", state="complete")
        except Exception:
            pass

    # Add assistant message to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})

    # Force refresh to update sidebar title after first exchange
    if len(st.session_state["message_history"]) <= 2:
        st.rerun()