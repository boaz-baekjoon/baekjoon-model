import numpy as np
import duckdb
from sklearn.metrics.pairwise import cosine_similarity

item_emb = np.load('data/item_embed.npy')
item_emb.shape # (item_emb, 40)

cosine_sim = cosine_similarity(item_emb, item_emb)

duck_db = duckdb.connect(database='database/recsys.db', read_only=True)
result = duck_db.execute(f"SELECT problem_id FROM problem_detail;").fetchall()
problem_list = [int(problem[0]) for problem in result] if result else None
duck_db.close()

problem_list = np.array(np.unique(problem_list))

with open("data/cosine_sim.npy", "wb") as f:
            np.save(f, cosine_sim)
            
with open("data/problem_list.npy", "wb") as f:
            np.save(f, problem_list)