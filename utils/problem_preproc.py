import duckdb
import psycopg2
import pandas as pd
import os

def load_problem_data():
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
    query = "SELECT problem_id, problem_lang, problem_level, tag_key FROM problem_details;"
    cur.execute(query)
    rows = cur.fetchall()
    
    # Writing results to duckdb
    db = duckdb.connect(database='database/recsys.db')
    db.execute(f'DROP TABLE IF EXISTS problem_detail;')
    db.execute("CREATE TABLE problem_detail (problem_id INTEGER, problem_lang VARCHAR, problem_level INTEGER, tag_key VARCHAR);")
    db.executemany("INSERT INTO problem_detail VALUES (?, ?, ?, ?)", rows)
    print("Data written to 'database/recsys.db'")
    
    # Close the cursor and connection
    cur.close()
    conn.close()

def preproc_problem():
    
    db = duckdb.connect(database='database/recsys.db', read_only=False)
    
    print("Map tag_key to tag_main using tag_key_to_11")
    db.execute("ALTER TABLE problem_detail ADD COLUMN category VARCHAR;")
    query = "UPDATE problem_detail SET category = CASE \
                WHEN tag_key IN ('implementation', 'arithmetic', 'case_work', 'ad_hoc', 'simulation', 'euclidean', 'arbitrary_precision', 'sorting', 'prefix_sum', 'recursion', 'pythagoras', 'precomputation', '2_sat', 'coordinate_compression', 'pigeonhole_principle', 'euler_characteristic', 'physics', 'statistics', 'slope_trick') THEN 'implementation' \
                WHEN tag_key IN ('data_structures', 'deque', 'stack', 'queue', 'linked_list', 'trie', 'segtree', 'tree_set', 'hash_set', 'bitmask', 'bitset', 'disjoint_set', 'priority_queue', 'two_pointer', 'splay_tree', 'rope', 'pst', 'hld', 'mo', 'rb_tree') THEN 'ds' \
                WHEN tag_key IN ('dp', 'dp_bitfield', 'dp_tree', 'dp_connection_profile', 'dp_deque', 'knapsack', 'lis', 'dp_sum_over_subsets', 'smaller_to_larger', 'dp_sum_over_subsets') THEN 'dp' \
                WHEN tag_key IN ('graphs', 'topological_sorting', 'graph_traversal', 'bfs', 'dfs', 'mst', 'scc', 'dijkstra', 'floyd_warshall', 'bellman_ford', 'flow', 'bipartite_matching', 'bipartite_graph', 'planar_graph', 'cactus', 'biconnected_component', 'dual_graph', 'dominator_tree', 'directed_mst', 'a_star') THEN 'graph' \
                WHEN tag_key IN ('binary_search', 'ternary_search', 'bruteforcing', 'backtracking', 'parametric_search', '0_1_bfs', 'bfs_01', 'mitm', 'bidirectional_search', 'burnside') THEN 'search' \
                WHEN tag_key IN ('regex', 'string', 'kmp', 'manacher', 'rabin_karp', 'suffix_array', 'suffix_tree', 'aho_corasick', 'palindrome_tree', 'hashing', 'knuth_x', 'manacher') THEN 'string' \
                WHEN tag_key IN ('math', 'number_theory', 'primality_test', 'sieve', 'combinatorics', 'probability', 'calculus', 'numerical_analysis', 'euler_phi', 'modular_multiplicative_inverse', 'crt', 'discrete_log', 'mobius_inversion', 'matrix_exponentiation', 'exponentiation_by_squaring', 'extended_euclidean', 'miller_rabin', 'pollard_rho', 'generating_function', 'discrete_kth_root', 'discrete_sqrt', 'knuth', 'hall', 'flt', 'matroid', 'circulation') THEN 'math' \
                WHEN tag_key IN ('greedy', 'divide_and_conquer', 'divide_and_conquer_optimization', 'linear_programming', 'duality', 'hungarian', 'gradient_descent', 'game_theory', 'majority_vote', 'alien', 'offline_queries', 'online_queries', 'randomization', 'constructive', 'simulated_annealing', 'cht', 'stable_marriage') THEN 'opt' \
                WHEN tag_key IN ('geometry', 'geometry_3d', 'convex_hull', 'line_intersection', 'sweeping', 'parsing', 'point_in_convex_polygon', 'polygon_area', 'voronoi', 'half_plane_intersection', 'rotating_calipers', 'pick', 'linearity_of_expectation', 'min_enclosing_circle', 'point_in_non_convex_polygon', 'delaunay', 'general_matching', 'centroid', 'centroid_decomposition', 'hirschberg') THEN 'geo' \
                WHEN tag_key IN ('fft', 'trees', 'mfmc', 'eulerian_path', 'lazyprop', 'lca', 'pbs', 'inclusion_and_exclusion', 'gaussian_elimination', 'linear_algebra', 'sprague_grundy', 'permutation_cycle_decomposition', 'articulation', 'sparse_table', 'tree_isomorphism', 'multi_segtree', 'link_cut_tree', 'top_tree', 'heuristics', 'degree_sequence', 'tsp', 'berlekamp_massey', 'kitamasa', 'cartesian_tree', 'polynomial_interpolation', 'birthday', 'functional_grap', 'dancing_links', 'merge_sort_tree', 'sqrt_decomposition', 'euler_tour_technique') THEN 'adv' \
                WHEN tag_key IN ('green', 'tree_decomposition', 'differential_cryptanalysis', 'geometric_boolean_operations', 'chordal_graph', 'utf8', 'lucas', 'geometry_hyper', 'bayes', 'offline_dynamic_connectivity', 'monotone_queue_optimization', 'stoer_wagner', 'hackenbush', 'z', 'suffix_tree', 'suffix_tree', 'alien', 'tree_compression', 'multipoint_evaluation', 'birthday', 'functional_graph', 'mcmf', 'sliding_window', 'euler_phi', 'general_math') THEN 'other' \
                ELSE NULL END;"
    db.execute(query)
    db.execute("CREATE INDEX problem_id_index ON problem_detail(problem_id);")
    print("Data written to 'database/recsys.db'")
    
    # save mapping table   
    data = {
    'category': ['implementation', 'ds', 'dp', 'graph', 'search', 'string', 'math', 'opt', 'geo', 'adv'],
    'category_int': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    df = pd.DataFrame(data)

    db.execute("DROP TABLE IF EXISTS category_mapping;")
    db.execute("""
    CREATE TABLE IF NOT EXISTS category_mapping (
        category VARCHAR,
        category_int INTEGER
    );
    """)
    
    db.executemany("INSERT INTO category_mapping VALUES (?, ?)", list(df.itertuples(index=False, name=None)))
    print("Data successfully saved to 'database/recsys.db'")
    
    db.close()
    
    
    
    
    
    