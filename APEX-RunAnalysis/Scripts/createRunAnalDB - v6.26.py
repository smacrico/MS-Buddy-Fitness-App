############### create RunningAnalysis database in Production environment
# dev 4.0 - fixed statments for training score and training log
# 
# 
# 
import sqlite3
import sys


def create_table_if_not_exists():
    conn = sqlite3.connect(r'c:/smakrykoDBs/Apex.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS running_sessions (
            running_economy INT, 
            date TXT, 
            distance INT, 
            sport TXT, 
            vo2max INT,  
            cardiacdrift INT,
            heart_rate INT,
            time INT,
            calories INT,
            avg_speed REAL,
            max_speed REAL,
            HR_RS_Deviation_Index INT
        )
    ''')
    
    conn.commit()
    conn.close()
    
create_table_if_not_exists()
    
try:
    # Establish connections to both databases
    conn_artemis = sqlite3.connect('c:/smakrykoDBs/artemis.db')
    conn_garmin = sqlite3.connect('c:/smakrykoDBs/garmin_activities.db')
    conn_running_analysis = sqlite3.connect('c:/smakrykoDBs/Apex.db')

    # Create cursors
    cursor_artemis = conn_artemis.cursor()
    cursor_garmin = conn_garmin.cursor()
    cursor_running_analysis = conn_running_analysis.cursor()
    
    # Select the specific columns from Artemis database joined with GarminDB activities
    cursor_artemis.execute('''
        SELECT 
            a.running_economy, 
            a.timestamp, 
            a.distance, 
            a.sport, 
            a.vo2maxsession,  
            a.cardiacdrift, 
            a.avg_heart_rate, 
            a.total_elapsed_time,
            g.avg_speed,
            g.max_speed,
            a.HR_RS_Deviation_Index
        FROM Artemistbl_fields a
        INNER JOIN activities g ON a.activity_id = g.activity_id
        WHERE a.sport like 'running'
    ''')

    # Fetch all the rows
    rows = cursor_artemis.fetchall()

    # Insert the data into running_session table in RunningAnalysis database
    cursor_running_analysis.executemany('''
        INSERT INTO running_sessions (running_economy, date, distance, sport, vo2max,  cardiacdrift, heart_rate, time, avg_speed, max_speed, HR_RS_Deviation_Index)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', rows)

    # Commit the changes
    conn_running_analysis.commit()

except sqlite3.Error as e:
    print(f"An error occurred: {e}")
    # Rollback any changes if an error occurs
    conn_running_analysis.rollback()

finally:
    # Always close the connections
    if conn_artemis:
        conn_artemis.close()
    if conn_garmin:
        conn_garmin.close()
    if conn_running_analysis:
        conn_running_analysis.close()

print("Data transfer completed successfully!")