import sqlite3
from datetime import datetime

# Test script to verify metrics_breakdown table
db_path = r'c:/smakrykoDBs/Apex.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics_breakdown'")
    result = cursor.fetchone()
    
    if result:
        print("✓ metrics_breakdown table exists")
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM metrics_breakdown")
        count = cursor.fetchone()[0]
        print(f"  Current row count: {count}")
        
        # Show all data
        cursor.execute("SELECT * FROM metrics_breakdown")
        rows = cursor.fetchall()
        if rows:
            print(f"\n  Found {len(rows)} row(s):")
            for row in rows:
                print(f"    Date: {row[0]}, Score: {row[1]}")
        else:
            print("  ⚠ Table is empty!")
            
        # Get column count
        cursor.execute("PRAGMA table_info(metrics_breakdown)")
        columns = cursor.fetchall()
        print(f"\n  Table has {len(columns)} columns:")
        for col in columns[:10]:  # Show first 10 columns
            print(f"    - {col[1]} ({col[2]})")
        if len(columns) > 10:
            print(f"    ... and {len(columns) - 10} more columns")
            
        # Try inserting a test record
        print("\n  Attempting test insert...")
        test_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Count columns in table
        placeholders = ','.join(['?' for _ in range(len(columns))])
        test_values = [test_date, 75.5] + [0.0] * (len(columns) - 2)
        
        cursor.execute(f"INSERT INTO metrics_breakdown VALUES ({placeholders})", test_values)
        conn.commit()
        
        # Verify insert
        cursor.execute("SELECT COUNT(*) FROM metrics_breakdown")
        new_count = cursor.fetchone()[0]
        print(f"  ✓ Test insert successful! New row count: {new_count}")
        
    else:
        print("✗ metrics_breakdown table does NOT exist!")
    
    conn.close()
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
