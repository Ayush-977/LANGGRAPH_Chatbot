import streamlit as st
import uuid
from langgraph_backend import chatbot, clear_database
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage,AIMessage
from langchain_groq import ChatGroq

from session_db import init_db, save_session_title, get_all_sessions, delete_all_sessions
init_db()

# --- 1. Helper Functions ---

def generate_title(first_message_content):
    """Generates a short title using LLM based on the first message."""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant")
        messages = [
            SystemMessage(content="Generate a very short title (max 4 words) for this chat based on the user's overall message. Do not use quotes."),
            HumanMessage(content=first_message_content)
        ]
        response = llm.invoke(messages)
        title = response.content.strip()
        if len(title) > 30:
            return title[:27] + "..."
        return title
    except Exception:
        return "New Conversation"

def load_convo(thread_id):
    """Loads message history from LangGraph state."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    return state.values.get("messages", [])

# --- 2. Session State Initialization ---

if "thread_id" not in st.session_state:
    # Default to a new random ID if none exists
    st.session_state["thread_id"] = str(uuid.uuid4())

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# --- 3. Sidebar (Persistent Menu Logic) ---

with st.sidebar:
    st.set_page_config(layout="wide")
    st.title("LangGraph Chat")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        save_session_title(new_id, "New Chat")
        st.session_state['thread_id'] = new_id
        st.session_state['message_history'] = []
        st.rerun()
    
    st.markdown("---")
    st.header("History")

    # 2. Load Sessions from DB (Persistent)
    # db_sessions is a list of tuples
    db_sessions = get_all_sessions()
    
    # Reverse so newest is at the top
    db_sessions = db_sessions[::-1] 
    
    # Extract IDs and Titles for the Radio button
    thread_ids = [s[0] for s in db_sessions]
    thread_titles = {s[0]: s[1] for s in db_sessions}

    # Ensure current thread is in the list (edge case for very first run)
    if st.session_state['thread_id'] not in thread_ids:
        thread_ids.insert(0, st.session_state['thread_id'])
        thread_titles[st.session_state['thread_id']] = "New Chat"

    # Find index of current thread
    try:
        curr_idx = thread_ids.index(st.session_state['thread_id'])
    except ValueError:
        curr_idx = 0

    # 3. Selection Menu
    selected_id = st.radio(
        "Select Chat",
        options=thread_ids,
        format_func=lambda x: thread_titles.get(x, "New Chat"),
        index=curr_idx,
        label_visibility="collapsed"
    )

    # 4. Handle Switching Chats
    if selected_id != st.session_state['thread_id']:
        st.session_state['thread_id'] = selected_id
        
        # Load messages from LangGraph backend
        messages = load_convo(selected_id)
        
        # Format for Streamlit UI
        temp_msgs = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_msgs.append({'role': role, 'content': msg.content})
            
        st.session_state['message_history'] = temp_msgs
        st.rerun()

    # --- STYLE INJECTION ---
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
        }
        /* Target the Clear History button specifically if possible, or last button */
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
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    #5. Clear History Button
    if st.button("⚠️ Clear History"):
        clear_database()
        delete_all_sessions()
        st.session_state.clear()
        st.rerun()

# --- 4. Main Chat Area ---

# Display Chat History
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input("Type here...")

if user_input:
    # We check if history is empty OR if title is still generic "New Chat"
    current_title = thread_titles.get(st.session_state['thread_id'], "New Chat")
    
    if len(st.session_state['message_history']) == 0 or current_title == "New Chat":
        # Generate new title
        new_name = generate_title(user_input)
        # Save to Database immediately
        save_session_title(st.session_state['thread_id'], new_name)

    # Add User Message to State
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        
        def stream_generator():
            for chunk, meta in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if chunk.content:
                    yield chunk.content
        
        ai_msg = st.write_stream(stream_generator)

    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Add AI Message to State
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_msg})

    # Force Refresh (Only on first message to update the sidebar title)
    if len(st.session_state['message_history']) <= 2:
        st.rerun()