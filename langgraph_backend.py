import sqlite3
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from datetime import datetime
from dotenv import load_dotenv
import requests
load_dotenv()

# --- Tools ---
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def eval_math(expression: str) -> dict:
    "Evaluate a math expression safely."
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def http_get(url: str) -> dict:
    "Send a GET request and return JSON or text."
    try:
        r = requests.get(url, timeout=10)
        try:
            return r.json()
        except:
            return {"text": r.text}
    except Exception as e:
        return {"error": str(e)}


@tool
def current_time(_) -> str:
    "Returns the current server time."
    return datetime.now().isoformat()


@tool
def python_eval(code: str) -> dict:
    "Execute pure Python (no imports)."
    safe_globals = {}
    try:
        exec(code, safe_globals)
        return safe_globals
    except Exception as e:
        return {"error": str(e)}


@tool
def read_file(path: str) -> str:
    "Read a text file."
    try:
        return open(path, "r").read()
    except Exception as e:
        return str(e)
    
@tool
def write_file(path: str, content: str) -> str:
    "Write content to a file."
    try:
        open(path, "w").write(content)
        return "File written successfully."
    except Exception as e:
        return str(e)



tools = [eval_math,search_tool,http_get,current_time,extract_keywords,python_eval,read_file,write_file]

# --- 1. Setup Database ---
DB_NAME = "chatbot.db"

conn = sqlite3.connect(database=DB_NAME, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# --- 2. Define State & Model ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_tools = llm.bind_tools(tools)

def chat(state: AgentState) -> AgentState:
    system_msg = SystemMessage(
    content=(
        "You are a helpful AI assistant.\n"
        "\n"
        "TOOL USE POLICY\n"
        "• Decide on your own when to use tools—do not wait for the user to ask.\n"
        "• Use tools for: (a) up-to-date or factual lookups, (b) calculations or data processing, "
        "(c) fetching or manipulating external resources, (d) verifying uncertain claims, "
        "or (e) any action that benefits from structured, reliable sources.\n"
        "• Answer directly from your own knowledge only when the question is simple, broadly known, "
        "and you are confident the information is accurate and not time-sensitive.\n"
        "• If you are uncertain whether your knowledge is sufficient, prefer using a tool to verify.\n"
        "• If the user's request is ambiguous, ask a brief, targeted clarifying question (do not ask whether they want tools—just ask for the missing detail).\n"
        "\n"
        "STYLE\n"
        "• Be concise and helpful. Show steps only when it aids understanding.\n"
        "• If a tool fails or returns an error, gracefully retry once if appropriate or explain the issue and provide alternatives.\n"
        "\n"
        "SAFETY & HALLUCINATION CONTROL\n"
        "• Do not fabricate facts, citations, or data. When uncertain, use tools or say that you are unsure and propose next steps.\n"
        "• For subjective questions, state that it is opinion-based and focus on balanced guidance.\n"
        "USE SEARCH TOOL WHEN YOU ARE NOT HAVE THE INFORMATION ABOUT THE USER'S QUERY."
    )
)
    messages = [system_msg] + state["messages"]
    
    response = llm_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  


def debug_tools_condition(state):
    out = tools_condition(state)
    # Normalize to a list for printing
    labels = out if isinstance(out, (list, tuple)) else [out]
    print("tools_condition returned labels:", labels)
    return labels


# --- 3. Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("llm", chat)
graph.add_node("tools",tool_node)
graph.add_edge(START, "llm")

graph.add_conditional_edges(
    "llm",
    debug_tools_condition,  
    {
        "tools": "tools",
        "end": END,
        "END": END,
        "stop": END,
        "__end__": END,
    },
)

graph.add_edge("tools","llm")

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