import sqlite3

DB_NAME = "chat_sessions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (thread_id TEXT PRIMARY KEY, title TEXT)''')
    conn.commit()
    conn.close()

def save_session_title(thread_id, title):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO sessions (thread_id, title) VALUES (?, ?)''', (thread_id, title))
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT thread_id, title FROM sessions')
    data = c.fetchall()
    conn.close()
    return data

def delete_all_sessions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM sessions')
    conn.commit()
    conn.close()