import sqlite3
import os

db_path = r"C:\smakrykoDBs\artemis.db"

print(f"Checking database at: {db_path}")
print(f"File exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"\nTables found: {len(tables)}")
    for table in tables:
        print(f"  - {table}")
    
    # Check if alert_logs exists and has the expected columns
    if 'alert_logs' in tables:
        cur.execute("PRAGMA table_info(alert_logs)")
        columns = cur.fetchall()
        print(f"\nalert_logs columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
    
    # Check if activities table exists
    if 'activities' in tables:
        cur.execute("SELECT COUNT(*) FROM activities")
        count = cur.fetchone()[0]
        print(f"\nactivities table has {count} rows")
    
    conn.close()
    print("\n✓ Database is accessible!")
else:
    print("\n✗ Database file not found!")
