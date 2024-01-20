import pandas as pd
import numpy as np
import duckdb
import psycopg2
import csv
import os


def load_seq_data():
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
    query = "SELECT * FROM user_sequence;"
    cur.execute(query)
    rows = cur.fetchall()

    # Writing results to a CSV file            
    with open(os.environ['DATA_PATH'] + '/user_sequence.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([i[0] for i in cur.description])  # Writing headers
        writer.writerows(rows)
    print("Data written to 'data/user_sequence.csv'")

    # Close the cursor and connection
    cur.close()
    conn.close()

def replace_comma1(seq):
    return seq.replace("'', ", "")

def replace_comma2(seq):
    return seq.replace(", '')", ")")

def split_seq(seq):
    if seq == '()' or seq != seq:
        return np.nan
    else:
        return [int(x[1:-1]) for x in seq[1:-1].split(", ")]

def preproc_seq():
    
    db = duckdb.connect('database/recsys.db')
    
    # load data and preprocess
    print("Load data and preprocess")
    seq_data = pd.read_csv("data/user_sequence.csv")
    seq_data["problem_sequence"] = seq_data["problem_sequence"].apply(replace_comma1)
    seq_data["problem_sequence"] = seq_data["problem_sequence"].apply(replace_comma2)
    seq_data['problem_sequence'] = seq_data["problem_sequence"].apply(split_seq)
    
    seq_data = seq_data[seq_data['problem_sequence'].notna()]

    # user_id map
    user_id_map = dict()
    for i, user_id in enumerate(seq_data['user_id'].unique()):
        user_id_map[user_id] = i
    seq_data['user_id'] = seq_data['user_id'].map(user_id_map)
    
    # save user_sequence
    seq_data.to_sql(con=db, name='user_sequence', if_exists='replace', index=False)
    print("user_sequence data written to 'database/recsys.db'")
    
    # save_user_id_map
    print("Save user_id_map")
    user_id_map_df = pd.DataFrame.from_dict(user_id_map, orient='index').reset_index()
    user_id_map_df.columns = ['user_id', 'user_id_int']
    
    db.execute(f'DROP TABLE IF EXISTS user_id_map;')
    db.execute("""CREATE TABLE IF NOT EXISTS user_id_map (
        user_id VARCHAR,
        user_id_int INTEGER,
        PRIMARY KEY (user_id)
        );""")
    
    user_id_map_df.to_sql(con=db, name='user_id_map', if_exists='replace', index=False)
    print("user_id_map data written to 'database/recsys.db'")

    db.close()

    # explode problem_sequence
    print("Explode problem_sequence")
    df = seq_data.explode('problem_sequence')
    df.to_csv("data/baekjoon.txt", header=False, index=False, sep=' ')
