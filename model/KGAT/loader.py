import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DataLoaderKGAT:
    def __init__(self, args) -> None:
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        # self.num_workers = args.num_workers # TODO

        # dataset
        """
        1. train.txt
        user_id item_ids
        0 5 6 8 10 ...
        1 3 5 7 22 ...
        
        2. text.txt
        0 5 6 8 10 ...
        1 3 5 7 22 ...
        
        3. kg_final.txt (header = None)
        h r t
        12700 0 48123   
        18104 0 48123
        
        """
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, "train.txt")
        self.test_file = os.path.join(self.data_dir, "test.txt")
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.statistic_cf()

        if self.use_pretrain == 1:
            self.load_pretrained_data()

        # ---------

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        # self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()
        
        self.print_info()

    def load_cf(self, filename) -> ((np.array, np.array), dict):
        print("loading cf... ", filename)
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, "r").readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        print("finish loading cf... ", filename)
        return (user, item), user_dict

    def statistic_cf(self):
        print("start statistics cf...")
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])  # user
        self.n_cf_test = len(self.cf_test_data[0])  # user
        print("finish statistics cf...")

    def load_kg(self, filename) -> pd.DataFrame:
        print("loading kg... ", filename)
        kg_data = pd.read_csv(filename, sep=" ", names=["h", "r", "t"], engine="python")
        kg_data = kg_data.drop_duplicates()
        print("finished loading kg... ", filename)
        return kg_data

    def construct_data(self, kg_data) -> None:
        print("constructing data...")
        # add inverse kg data
        n_relations = max(kg_data["r"]) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({"h": "t", "t": "h"}, axis="columns")
        inverse_kg_data["r"] += n_relations  # 반대 방향의 relation을 이어서 정의한 것.
        kg_data = pd.concat(
            [kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False
        )

        # re-map user id -> 추후에는 전처리 단계에서 처리하자!
        kg_data["r"] += 2
        self.n_relations = max(kg_data["r"]) + 1
        self.n_entities = max(max(kg_data["h"]), max(kg_data["t"])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(
                list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))
            ).astype(np.int32),
            self.cf_train_data[1].astype(np.int32),
        )
        self.cf_test_data = (
            np.array(
                list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))
            ).astype(np.int32),
            self.cf_test_data[1].astype(np.int32),
        )

        self.train_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.train_user_dict.items()
        }
        self.test_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.test_user_dict.items()
        }

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        cf2kg_train_data["h"] = self.cf_train_data[0]
        cf2kg_train_data["t"] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        inverse_cf2kg_train_data["h"] = self.cf_train_data[1]
        inverse_cf2kg_train_data["t"] = self.cf_train_data[0]

        self.kg_train_data = pd.concat(
            [kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True
        )
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

        print("finished constructing data...")

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def create_adjacency_dict(self):
        print("creating adj mat...")
        self.adjacency_dict = {}
        for (
            r,
            ht_list,
        ) in self.train_relation_dict.items():  # relation별 adjancency_matrix 구성
            rows = [e[0] for e in ht_list]  # h
            cols = [e[1] for e in ht_list]  # t
            vals = [1] * len(rows)
            adj = sp.coo_matrix(
                (vals, (rows, cols)),
                shape=(self.n_users_entities, self.n_users_entities),
            )
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        print("creating lap mat...")

        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == "symmetric":
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == "random-walk":
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())  # 희소행렬 처리

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(
        self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx
    ):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(
                kg_dict, h, relation[0], 1, highest_neg_idx
            )
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def load_pretrained_data(self):
        pretrain_path = "%s/%s/%s.npz" % (
            self.pretrain_embedding_dir,
            self.data_name,
            self.args.premodel,
        )
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data["user_embed"]
        self.item_pre_embed = pretrain_data["item_embed"]

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim

    def print_info(self):
        print('n_users:           %d' % self.n_users)
        print('n_items:           %d' % self.n_items)
        print('n_entities:        %d' % self.n_entities)
        print('n_users_entities:  %d' % self.n_users_entities)
        print('n_relations:       %d' % self.n_relations)

        print('n_h_list:          %d' % len(self.h_list))
        print('n_t_list:          %d' % len(self.t_list))
        print('n_r_list:          %d' % len(self.r_list))

        print('n_cf_train:        %d' % self.n_cf_train)
        print('n_cf_test:         %d' % self.n_cf_test)

        print('n_kg_train:        %d' % self.n_kg_train)
