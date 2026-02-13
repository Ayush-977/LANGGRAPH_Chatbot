import uuid
import streamlit as st

from langgraph_backend import chatbot, clear_database
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq

from session_db import init_db, save_session_title, get_all_sessions, delete_all_sessions

# ----------------
# Initialization
# ----------------
init_db()


st.set_page_config(layout="wide")


# ---------------------
# 1) Helper Functions
# ---------------------

@st.cache_resource
def get_title_llm():
    """Cache the LLM client used for generating titles."""
    return ChatGroq(model="llama-3.1-8b-instant")


def generate_title(first_message_content: str) -> str:
    """Generate a very short, safe title (≤ 4 words, ≤ 30 chars) for the chat."""
    try:
        llm = get_title_llm()
        messages = [
            SystemMessage(content="Return ONLY a concise chat title, MAX 4 words, no quotes."),
            HumanMessage(content=first_message_content),
        ]
        response = llm.invoke(messages)
        title = (response.content or "").strip()
        title = " ".join(title.split())

        words = title.split()
        if len(words) > 4:
            title = " ".join(words[:4])

        if len(title) > 30:
            title = title[:27] + "…"

        return title or "New Conversation"
    except Exception:
        return "New Conversation"


def load_convo(thread_id: str):
    """
    Load message history from LangGraph state and return a list of dicts
    formatted for Streamlit chat UI. Only include Human/AI messages.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])

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


# -----------
# 3) Sidebar 
# -----------
with st.sidebar:
    st.title("LangGraph Chat")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        save_session_title(new_id, "New Chat")
        st.session_state["thread_id"] = new_id
        st.session_state["message_history"] = []
        st.rerun()

    st.markdown("---")
    st.header("History")

    # Load Sessions from DB 
    db_sessions = get_all_sessions()
    db_sessions = db_sessions[::-1]

    # Extract IDs and Titles
    thread_ids = [s[0] for s in db_sessions]
    thread_titles = {s[0]: s[1] for s in db_sessions}

    if st.session_state["thread_id"] not in thread_ids:
        thread_ids.insert(0, st.session_state["thread_id"])
        thread_titles[st.session_state["thread_id"]] = "New Chat"

    try:
        curr_idx = thread_ids.index(st.session_state["thread_id"])
    except ValueError:
        curr_idx = 0

    # Selection Menu
    selected_id = st.radio(
        "Select Chat",
        options=thread_ids,
        format_func=lambda x: thread_titles.get(x, "New Chat"),
        index=curr_idx,
        label_visibility="collapsed",
    )

    # Handle Switching Chats
    if selected_id != st.session_state["thread_id"]:
        st.session_state["thread_id"] = selected_id
        st.session_state["message_history"] = load_convo(selected_id)
        st.rerun()

    # --- Style Injection ---
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            width: 100%;
        }
        /* Heuristic styling for the last button (Clear History) */
        div.stButton > button:last-child {
            background-color: #ff4b4b;
            color: white;
            border: none;
        }
        div.stButton > button:last-child:hover {
            background-color: #ff0000;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # 3.5 Clear History Button
    if st.button("⚠️ Clear History"):
        clear_database()
        delete_all_sessions()
        st.session_state.clear()
        st.rerun()


# -------------------
# 4) Main Chat Area
# -------------------

# Display Chat History
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # -- Title generation --
    current_title = thread_titles.get(st.session_state["thread_id"], "New Chat")
    if len(st.session_state["message_history"]) == 0 or current_title == "New Chat":
        new_name = generate_title(user_input)
        save_session_title(st.session_state["thread_id"], new_name)

    # -- Add User Message to State and UI
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        status_box = {"ref": None} 

        def unified_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_box["ref"] is None:
                        status_box["ref"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_box["ref"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        assistant_text = st.write_stream(unified_stream())

        if status_box["ref"] is not None:
            status_box["ref"].update(label="✅ Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": assistant_text})

    if len(st.session_state["message_history"]) <= 2:
        st.rerun()