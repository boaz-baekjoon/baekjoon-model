import duckdb
import psycopg2
import os

def load_user_data():
    # Establishing the connection
    conn = psycopg2.connect(
        host=os.environ["HOST"],
        port=os.environ["PORT"],
        database=os.environ["DATABASE"],
        user=os.environ["POSTGRE_USER"],
        password=os.environ["PASSWORD"],
    )
    print("Connection established")

    
    cur = conn.cursor()
    query = "SELECT user_id, user_tier FROM user_details;"
    cur.execute(query)
    rows = cur.fetchall()
    
    # Writing results to duckdb
    db = duckdb.connect(database='database/recsys.db')
    db.execute(f'DROP TABLE IF EXISTS user_details;')
    db.execute("CREATE TABLE user_details (user_id VARCHAR, user_tier INTEGER);")
    db.executemany("INSERT INTO user_details VALUES (?, ?)", rows)
    print("Data written to 'database/recsys.db'")
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    