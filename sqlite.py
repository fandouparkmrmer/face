import sqlite3

# 初始化数据库
def init_db():
    conn = sqlite3.connect('blacklist.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS blacklist
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# 插入一些失信人员数据
def add_to_blacklist(name):
    conn = sqlite3.connect('blacklist.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO blacklist (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

# 删除失信人员数据
def remove_from_blacklist(name):
    conn = sqlite3.connect('blacklist.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM blacklist WHERE name = ?", (name,))
    conn.commit()
    conn.close()

def is_blacklisted(name):
    conn = sqlite3.connect('blacklist.db')
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM blacklist WHERE name = ?", (name,))
    exists = cursor.fetchone()
    conn.close()
    return exists is not None