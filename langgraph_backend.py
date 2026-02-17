import sqlite3
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage

from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()


# --- 1. Setup Database ---
DB_NAME = "chatbot.db"

conn = sqlite3.connect(database=DB_NAME, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# --- 2. Define State & Model ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")


def chat(state: AgentState) -> AgentState:
    system_msg = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use tools only when the user explicitly asks for information "
            "that requires calling a tool. "
            "For greetings, casual chat, or explanations, respond normally "
            "without calling any tool. "
            "Think step-by-step and choose the simplest action."
        )
    )
    messages = [system_msg] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

    

# --- 3. Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("llm", chat)
graph.add_edge(START, "llm")
graph.add_edge("llm",END)

chatbot = graph.compile(checkpointer=checkpointer)

# --- 4. Helper Functions ---

def retrive_all_threads():
    """
    Directly queries the DB for unique thread_ids.
    Efficient O(1) lookup rather than loading all history.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []

def clear_database():
    """
    Wipes all persistence data from the SQLite database.
    """
    try:
        cursor = conn.cursor()
        # Delete LangGraph internal tables
        cursor.execute("DELETE FROM checkpoints")
        cursor.execute("DELETE FROM checkpoint_blobs")
        cursor.execute("DELETE FROM checkpoint_writes")
        conn.commit()
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")