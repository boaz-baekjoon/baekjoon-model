import numpy as np
import duckdb
from ast import literal_eval


def query_user_id_map(user_id):
    duck_db = duckdb.connect(database='database/recsys.db', read_only=True)
    result = duck_db.execute(f"SELECT user_id_int FROM user_id_map WHERE user_id = '{user_id}';").fetchall()
    user_id_int = int(result[0][0]) if result else None
    duck_db.close()
    return user_id_int

def query_user_seq(user_id_int):
    duck_db = duckdb.connect(database='database/recsys.db', read_only=True)
    result = duck_db.execute(f"SELECT problem_sequence FROM user_sequence WHERE user_id = {user_id_int};").fetchall()
    user_seq = result[0][0] if result else None
    duck_db.close()
    return user_seq

def query_user_tier(user_id):
    duck_db = duckdb.connect(database='database/recsys.db', read_only=True)
    user_tier = int(duck_db.execute(f"SELECT user_tier FROM user_details WHERE user_id = '{user_id}';").fetchone()[0])
    duck_db.close()
    return user_tier

def query_problem_list(lower_tier, upper_tier, category_int, use_tier_and_cat=False):
    duck_db = duckdb.connect(database='database/recsys.db', read_only=True)
    result = duck_db.execute(f"SELECT problem_id FROM problem_detail WHERE problem_level >= {lower_tier} AND problem_level <= {upper_tier};").fetchall()
    problem_list = [int(problem[0]) for problem in result] if result else None
    if category_int:
        category = duck_db.execute(f"SELECT category FROM category_mapping WHERE category_int = {category_int};").fetchone()[0]
        result = duck_db.execute(f"SELECT problem_id FROM problem_detail WHERE category = '{category}';").fetchall()
        problem_list = [int(problem[0]) for problem in result] if result else None
    if use_tier_and_cat:
        category = duck_db.execute(f"SELECT category FROM category_mapping WHERE category_int = {category_int};").fetchone()[0]
        result = duck_db.execute(f"SELECT problem_id FROM problem_detail WHERE problem_level >= {lower_tier} AND problem_level <= {upper_tier} AND category = '{category}';").fetchall()
        problem_list = [int(problem[0]) for problem in result] if result else None
    duck_db.close()
    return problem_list

cosine_sim = np.load('data/cosine_sim.npy')
unique_problem_list = np.load('data/problem_list.npy').tolist()