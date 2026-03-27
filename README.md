# 🤖 LangGraph Chatbot with Groq & Streamlit

A stateful, high‑performance conversational AI chatbot powered by **LangGraph** for orchestration and **Groq** for ultra‑fast inference using Llama 3.  
This project includes a fully persistent conversation memory, Streamlit UI, SQLite-based session storage, and modular backend architecture.

🔗 **Live Demo:** https://chat-graph.streamlit.app/

---

## ✨ Features

### ⚡ Groq‑Accelerated Llama 3

Uses **Groq’s ultra‑fast LLM inference** to generate responses in real time.

### 🧠 Stateful Memory with LangGraph

Conversation history persists across turns using LangGraph’s state graph and SQLite checkpointing.

### 💾 Persistent Database Storage

All chat sessions, titles, and messages are stored in SQLite for retrieval across app restarts.

### 🎛️ LangGraph Architecture

A graph‑based workflow handles:

- LLM responses
- Tool calling
- State transitions
- Multi-step conversational logic

### 🎨 Streamlit Frontend

A clean and responsive UI featuring:

- Sidebar chat history
- Tool execution status
- Live streaming responses
- Auto‑generated conversation titles

### 🧱 Modular Codebase

- `langgraph_frontend.py` → Streamlit UI
- `langgraph_backend.py` → LLM, LangGraph nodes, tools
- `session_db.py` → SQLite session manager

---

## 🛠️ Tech Stack

| Component               | Technology     |
| ----------------------- | -------------- |
| **Language**            | Python 3.10+   |
| **LLM Framework**       | LangChain      |
| **Graph Orchestration** | LangGraph      |
| **Model Provider**      | Groq (Llama 3) |
| **Frontend**            | Streamlit      |
| **Database**            | SQLite         |

---

## 🚀 Installation & Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ayush-977/LANGGRAPH_Chatbot.git
cd LANGGRAPH_Chatbot
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Groq API Key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 5️⃣ Run the Application

```bash
streamlit run langgraph_frontend.py
```

---

## 📁 Project Structure

```
LANGGRAPH_Chatbot/
│
├── langgraph_frontend.py     # Streamlit UI
├── langgraph_backend.py      # LLM, LangGraph nodes, tools
├── session_db.py             # SQLite session storage
├── chatbot.db                # Auto-generated database
├── requirements.txt
├── README.md
└── .env
```

---

## 🧩 How It Works (High-Level Architecture)

### LangGraph Backend

- Defines **state machine** with `AgentState`
- Adds LLM node
- Supports tool execution (optional)
- Stores state using SQLite checkpointing

### Streamlit Frontend

- Displays chat messages
- Streams responses in real-time
- Auto-generates conversation titles
- Manages multiple conversations with radio menu
- Shows tool execution status using `st.status()`

### Registered Tools

- eval_math
- search_tool
- http_get
- wikipedia_search
- current_time
- extract_keywords
- python_eval
- read_file
- write_file

### Database

Stores:

- User sessions
- Titles
- Message history
- LangGraph checkpoints

---

## 🧰 Customizing the Chatbot

### 🟣 System Prompt

Defined in `langgraph_backend.py`:

```python
SystemMessage(
    content="You are a helpful assistant..."
)
```

### 🧰 Adding New Tools

```python
@tool
def my_custom_tool(input: str):
    return {"output": input.upper()}
```

Add to the list:

```python
tools = [ ..., my_custom_tool ]
```

### 🔵 Modify Frontend UI

Located in `langgraph_frontend.py`.

---

## 🧪 Example Usage

Ask the bot to:

- Search web
- Do math
- Fetch APIs
- Run Python code
- Extract keywords
- Read/write files

---

## 👨‍💻 Contributing

Pull requests are welcome!  
If you want to contribute:

1. Fork the repo
2. Create a feature branch
3. Submit a PR

---

## 📜 License

This project is open-source under the **MIT License**.
