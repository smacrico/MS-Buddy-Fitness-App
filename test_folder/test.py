import sqlite3
conn = sqlite3.connect(r'c:/users/jheel/jheelHealthData/DBs/garmin_activities.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(tables)  # This will list all tables in the database

conn.close()