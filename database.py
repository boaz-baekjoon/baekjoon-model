import pandas as pd

# data_load
problem_list = pd.read_csv("data/preproc_data/problem_id.csv")
user_problem_table = pd.read_csv('data/preproc_data/baekjoon.csv')
user_id_map = pd.read_csv("data/preproc_data/user_id_int_map.csv")
user_id_dict = user_id_map.set_index('user_id').to_dict()['user_id_int']
problem_detail = pd.read_csv("data/preproc_data/problem_detail_new.csv")[['problem_id', 'problem_level', 'category']]
user_detail = pd.read_csv("data/preproc_data/users_detail.csv")[['user_id','user_tier']]
category = pd.read_csv("data/preproc_data/category_id_int_map.csv")
category_dict = category.set_index('category_int').to_dict()['category']
