import sqlite3

db_path = r"C:\smakrykoDBs\artemis.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cur.fetchall()]
print("Tables in artemis.db:")
for table in tables:
    print(f"  - {table}")
conn.close()
