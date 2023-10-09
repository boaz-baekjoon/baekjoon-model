import pandas as pd

# data_load
problem_list = pd.read_csv("data/preproc_data/problem_id.csv")
user_problem_table = pd.read_csv('data/preproc_data/baekjoon.csv')
user_id_map = pd.read_csv("data/preproc_data/user_id_int_map.csv")
user_id_dict = user_id_map.set_index('user_id').to_dict()['user_id_int']