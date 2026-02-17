import uuid
import streamlit as st
import logging

from langgraph_backend import chatbot, clear_database
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq

from session_db import init_db, save_session_title, get_all_sessions, delete_all_sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------
# Initialization
# ----------------
init_db()

st.set_page_config(layout="wide")

# ---------------------
# 1) Helper Functions
# ---------------------

# Fallback singleton to avoid pickling issues with cache
_title_llm = None
def get_title_llm():
    global _title_llm
    if _title_llm is None:
        _title_llm = ChatGroq(model="llama-3.1-8b-instant")
    return _title_llm

def generate_title(first_message_content: str) -> str:
    """Generate a very short, safe title (≤ 4 words, ≤ 30 chars) for the chat."""
    try:
        llm = get_title_llm()
        messages = [
            SystemMessage(content="Return ONLY a concise chat title, MAX 4 words, no quotes."),
            HumanMessage(content=first_message_content),
        ]
        response = llm.invoke(messages)
        title = str(getattr(response, "content", "") or "").strip()
        title = " ".join(title.split())

        words = title.split()
        if len(words) > 4:
            title = " ".join(words[:4])

        if len(title) > 30:
            title = title[:27] + "…"

        return title or "New Conversation"
    except Exception as e:
        logger.exception("Title generation failed: %s", e)
        return "New Conversation"

def load_convo(thread_id: str):
    """Load message history from LangGraph state and return UI-ready list."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config) or {}
    except Exception as e:
        logger.warning("get_state failed for %s: %s", thread_id, e)
        return []

    values = getattr(state, "values", {}) or {}
    messages = values.get("messages", []) or []

    ui_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            ui_msgs.append({"role": "assistant", "content": msg.content})
    return ui_msgs

# ---------------------------------
# 2) Session State Initialization
# ---------------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Safe default to avoid NameError if sidebar fails
thread_titles = {st.session_state["thread_id"]: "New Chat"}

# -----------
# 3) Sidebar 
# -----------
with st.sidebar:
    st.title("LangGraph Chat")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        try:
            save_session_title(new_id, "New Chat")
        except Exception as e:
            logger.exception("Failed to save new session title: %s", e)
        st.session_state["thread_id"] = new_id
        st.session_state["message_history"] = []
        st.rerun()

    st.markdown("---")
    st.header("History")

    try:
        db_sessions = get_all_sessions()
    except Exception as e:
        logger.exception("get_all_sessions failed: %s", e)
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
        except Exception as e:
            logger.exception("clear_database failed: %s", e)
        try:
            delete_all_sessions()
        except Exception as e:
            logger.exception("delete_all_sessions failed: %s", e)
        st.session_state.clear()
        st.rerun()

# -------------------
# 4) Main Chat Area
# -------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    current_title = thread_titles.get(st.session_state["thread_id"], "New Chat")
    if len(st.session_state["message_history"]) == 0 or current_title == "New Chat":
        new_name = generate_title(user_input)
        try:
            save_session_title(st.session_state["thread_id"], new_name)
        except Exception as e:
            logger.exception("Failed to save session title: %s", e)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
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
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", None) or getattr(message_chunk, "tool_name", None) or "tool"
                    set_status(f"Using `{tool_name}` …", state="running")
                elif isinstance(message_chunk, AIMessage):
                    if isinstance(message_chunk.content, str) and message_chunk.content:
                        yield message_chunk.content

        ai_msg = st.write_stream(unified_stream())
        set_status("Tool finished", state="complete")

    st.session_state["message_history"].append({"role": "assistant", "content": ai_msg})

    if len(st.session_state["message_history"]) <= 2:
        st.rerun()