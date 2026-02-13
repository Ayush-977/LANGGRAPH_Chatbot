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
from dotenv import load_dotenv
import requests

load_dotenv()

# --- Tools ---
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(a:float,b:float,operation:str) -> dict:
  """
  Perform a basic arithmetic operation on two numbers.
  Supported operations: add,sub,mul,div
  """
  try:
    if operation == "add":
      result = a + b
    elif operation == "sub":
      result = a - b
    elif operation == "mul":
      result = a * b
    elif operation == "div":
      if b == 0:
        return {"error": "Division by zero is not possible"}
      result = a / b
    else:
      return {"error": f"Unsupported operation '{operation}'"}
    
    return {"a": a , "b": b , "operation":operation, "result":result}
  
  except Exception as e:
    return {"error": str(e)}



# --- 1. Setup Database ---
DB_NAME = "chatbot.db"

conn = sqlite3.connect(database=DB_NAME, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# --- 2. Define State & Model ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")
tools = [calculator,search_tool]

llm_tools = llm.bind_tools(tools)

def chat(state: AgentState) -> AgentState:
    return {"messages": [llm_tools.invoke([SystemMessage(content="You are a helpful assistant. Use tools only when required and give answer according to that tool output."), *state["messages"]])
]}

tool_node = ToolNode(tools)

# --- 3. Build Graph ---
graph = StateGraph(AgentState)
graph.add_node("llm", chat)
graph.add_node("tools",tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm",tools_condition)
graph.add_edge("llm", END)
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