import pandas as pd
import numpy as np
import random

data = pd.read_csv("../data/user_results.csv")

def trans_col(data:list):
        try:
            if data != data:
                col1 = list()
            else:
                col1 = [int(x[1:-1])for x in data.strip("[]").split(", ")]
                col1 = list(np.random.choice(col1, len(col1), replace=False)) # 문제 순서 섞기 (랜덤:변경 가능)
        except:
            print(data)
        return col1

# 문제 전처리
data['맞은 문제'] = data['맞은 문제'].apply(trans_col)
data['맞았지만 만점을 받지 못한 문제'] = data['맞았지만 만점을 받지 못한 문제'].apply(trans_col)
data['시도했지만 맞지 못한 문제'] = data['시도했지만 맞지 못한 문제'].apply(trans_col)
data['attempted'] = data['맞은 문제'] + data['맞았지만 만점을 받지 못한 문제'] + data['시도했지만 맞지 못한 문제']
data['correct'] = data['맞은 문제'] 
data['half_correct'] = data['맞았지만 만점을 받지 못한 문제']
data['wrong'] = data['시도했지만 맞지 못한 문제']

data = data[['user_id','attempted', 'correct','half_correct','wrong']]

def for_stats(data:list):
    return len(data)

print(min(data['attempted'].apply(for_stats)))
print(min(data['correct'].apply(for_stats)))
print(min(data['half_correct'].apply(for_stats)))
print(min(data['wrong'].apply(for_stats)))

data = data[(data['attempted'].apply(lambda x: len(x)) > 5)] # interaction이 5 이하인 사람 제거 -> 1명

# user_id map int type id
user_id_to_int = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
user_id_to_int_table = pd.DataFrame(list(user_id_to_int.items()), columns=['user_id', 'user_id_int'])
data['user_id_int'] = data['user_id'].map(user_id_to_int)
user_id_to_int_table.to_csv('./data/user_id_map.csv',index=False)

# train, test set split
# ver1. interact만을 고려한 모델
# todo : 틀린 문제 맞은 문제를 고려한 세팅

def train_test_split(data:list, train_prob = 0.1):
    test_n = int(len(data) * train_prob)
    test = list(set(random.sample(data, test_n)))
    train = list(set(data) - set(test))
    return train, test

split_data = data['attempted'].apply(train_test_split)

data['train'] = [train for train, _ in split_data]
data['test'] = [test for _, test in split_data]

def train_format_data(row):
    formatted = str(row['user_id_int']) + ' ' + ' '.join([f"{item}" for  item in row['train']])
    return formatted

def test_format_data(row):
    formatted = str(row['user_id_int']) + ' ' + ' '.join([f"{item}" for  item in row['test']])
    return formatted

formatted_data = data.apply(train_format_data, axis=1)
formatted_data_string = '\n'.join(formatted_data)

with open('./data/train.txt', 'w') as file:
    file.write(formatted_data_string)
    
formatted_data = data.apply(test_format_data, axis=1)
formatted_data_string = '\n'.join(formatted_data)

with open('./data/test.txt', 'w') as file:
    file.write(formatted_data_string)



# tag preproc
def tag_key_to_11(key):
    if key in ['implementation', 'arithmetic', 'case_work', 'ad_hoc', 'simulation', 'euclidean', 'arbitrary_precision', 'sorting', 'prefix_sum', 'recursion', 'pythagoras', 'precomputation', '2_sat', 'coordinate_compression', 'pigeonhole_principle', 'euler_characteristic', 'physics', 'statistics', 'slope_trick'
]: return 'Basic'
    if key in ['data_structures', 'deque', 'stack', 'queue', 'linked_list', 'trie', 'segtree', 'tree_set', 'hash_set', 'bitmask', 'bitset', 'disjoint_set', 'priority_queue', 'two_pointer', 'splay_tree', 'rope', 'pst', 'hld', 'mo', 'rb_tree'
]: return 'DS'
    if key in ['dp', 'dp_bitfield', 'dp_tree', 'dp_connection_profile', 'dp_deque', 'knapsack', 'lis', 'dp_sum_over_subsets', 'smaller_to_larger', 'dp_sum_over_subsets'
]: return 'DP'
    if key in ['graphs', 'topological_sorting', 'graph_traversal', 'bfs', 'dfs', 'mst', 'scc', 'dijkstra', 'floyd_warshall', 'bellman_ford', 'flow', 'bipartite_matching', 'bipartite_graph', 'planar_graph', 'cactus', 'biconnected_component', 'dual_graph', 'dominator_tree', 'directed_mst', 'a_star'
]: return 'Graph'
    if key in ['binary_search', 'ternary_search', 'bruteforcing', 'backtracking', 'parametric_search', '0_1_bfs', 'bfs_01', 'mitm', 'bidirectional_search', 'burnside'
]: return 'Search'
    if key in ['regex', 'string', 'kmp', 'manacher', 'rabin_karp', 'suffix_array', 'suffix_tree', 'aho_corasick', 'palindrome_tree', 'hashing', 'knuth_x', 'manacher'
]: return 'Str'
    if key in ['math', 'number_theory', 'primality_test', 'sieve', 'combinatorics', 'probability', 'calculus', 'numerical_analysis', 'euler_phi', 'modular_multiplicative_inverse', 'crt', 'discrete_log', 'mobius_inversion', 'matrix_exponentiation', 'exponentiation_by_squaring', 'extended_euclidean', 'miller_rabin', 'pollard_rho', 'generating_function', 'discrete_kth_root', 'discrete_sqrt', 'knuth', 'hall', 'flt', 'matroid', 'circulation'
]: return 'Math'
    if key in ['greedy', 'divide_and_conquer', 'divide_and_conquer_optimization', 'linear_programming', 'duality', 'hungarian', 'gradient_descent', 'game_theory', 'majority_vote', 'alien', 'offline_queries', 'online_queries', 'randomization', 'constructive', 'simulated_annealing', 'cht', 'stable_marriage'
]: return 'Opt'
    if key in ['geometry', 'geometry_3d', 'convex_hull', 'line_intersection', 'sweeping', 'parsing', 'point_in_convex_polygon', 'polygon_area', 'voronoi', 'half_plane_intersection', 'rotating_calipers', 'pick', 'linearity_of_expectation', 'min_enclosing_circle', 'point_in_non_convex_polygon', 'delaunay', 'general_matching', 'centroid', 'centroid_decomposition', 'hirschberg'
]: return 'Geo'
    if key in ['fft', 'trees', 'mfmc', 'eulerian_path', 'lazyprop', 'lca', 'pbs', 'inclusion_and_exclusion', 'gaussian_elimination', 'linear_algebra', 'sprague_grundy', 'permutation_cycle_decomposition', 'articulation', 'sparse_table', 'tree_isomorphism', 'multi_segtree', 'link_cut_tree', 'top_tree', 'heuristics', 'degree_sequence', 'tsp', 'berlekamp_massey', 'kitamasa', 'cartesian_tree', 'polynomial_interpolation', 'birthday', 'functional_grap', 'dancing_links', 'merge_sort_tree', 'sqrt_decomposition', 'euler_tour_technique'
]: return 'Adv'
    if key in ['green', 'tree_decomposition', 'differential_cryptanalysis', 'geometric_boolean_operations', 'chordal_graph', 'utf8', 'lucas', 'geometry_hyper', 'bayes', 'offline_dynamic_connectivity', 'monotone_queue_optimization', 'stoer_wagner', 'hackenbush', 'z', 'suffix_tree', 'suffix_tree', 'alien', 'tree_compression', 'multipoint_evaluation', 'birthday', 'functional_graph', 'mcmf', 'sliding_window', 'euler_phi', 'general_math'
]: return 'Other'
    else: return np.NaN



# KG graph
problem = pd.read_csv('../data/problem_detail.csv')
problem['tag_main'] = problem['tag_key'].apply(tag_key_to_11)

problem[['problem_id', 'tag_key', 'problem_level']]

max(problem['problem_id']) # 27999
len((problem[~problem['tag_key'].isna()]['tag_key']).unique()) # 198
len((problem[~problem['tag_main'].isna()]['tag_main']).unique()) # 11
len((problem[~problem['problem_level'].isna()]['problem_level']).unique()) # 31

level_map = {level : level + 27999 for level in sorted(problem['problem_level'].unique())}
tag_key_main_map = {tag: int(idx + 27999 + 31) for idx, tag in enumerate(problem[~problem['tag_main'].isna()]['tag_main'].unique())}
tag_key_middle_map = {tag: int(idx + 27999 + 31 + 11) for idx, tag in enumerate(problem[~problem['tag_key'].isna()]['tag_key'].unique())} # 31 - 229

entity_table = pd.DataFrame(list(level_map.items())+list(tag_key_main_map.items())+list(tag_key_middle_map.items()), columns = ['entity', 'entity_id'])
entity_table.to_csv('./data/entity_map.csv', index=False)

relation_map = {'Level':0, 'Tag:Main':1, 'Tag:Middle':2}
relation_map_table = pd.DataFrame(list(relation_map.items()), columns = ['relation', 'relation_id'])
relation_map_table.to_csv('./data/relation_map.csv', index=False)

problem['middle_idx'] = problem['tag_key'].map(tag_key_middle_map)
problem['main_idx'] = problem['tag_main'].map(tag_key_main_map)

def relation_level_format(row):
    formatted = str(row['problem_id']) + ' 0 ' + str(row['problem_level'] + 27999)
    return formatted

def relation_middle_tag_format(row):
    formatted = str(row['problem_id']) + ' 2 ' + str(int(row['middle_idx']))
    return formatted

def relation_main_tag_format(row):
    formatted = str(row['problem_id']) + ' 1 ' + str(int(row['main_idx']))
    return formatted


formatted_data_level  = problem.apply(relation_level_format, axis=1)
#formatted_data_middle = problem[~problem.tag_key.isna()].apply(relation_middle_tag_format, axis=1)
formatted_data_main = problem[~problem.tag_main.isna()].apply(relation_main_tag_format, axis=1)

#formatted_data_string = '\n'.join(formatted_data_level) + '\n' + '\n'.join(formatted_data_middle) + '\n' + '\n'.join(formatted_data_main)
formatted_data_string = '\n'.join(formatted_data_level) + '\n' + '\n'.join(formatted_data_main)

with open('./data/kg_final.txt', 'w') as file:
    file.write(formatted_data_string)