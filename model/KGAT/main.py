import sys
import os
import argparse
import random
from time import time

import pickle
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler, autocast

from model import *
from utils import *
from loader import *

def parse_kgat_args(opts=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2019, help="Random seed.")
    parser.add_argument("--data_name", nargs="?", default="baekjun", help="Choose a dataset from {yelp2018, last-fm, last-fm}")
    parser.add_argument("--data_dir", nargs="?", default="./data/", help="Input data path.")

    parser.add_argument("--loader_pickle", type=str, default="dataloader_new.pkl", help="none: no pickle, path/to/pickle_file.pkl: the pickle file to use from")
    parser.add_argument("--use_pretrain", type=int, default=0, help="0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.",)
    parser.add_argument("--pretrain_embedding_dir", nargs="?", default="datasets/pretrain/", help="Path of learned embeddings.",)
    parser.add_argument("--pretrain_model_path", nargs="?", default="trained_model/model.pth", help="Path of stored model.",)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cf_batch_size", type=int, default=256, help="CF batch size.")
    parser.add_argument("--kg_batch_size", type=int, default=256, help="KG batch size.")
    parser.add_argument("--test_batch_size", type=int, default=5000,  # 10000 help="Test batch size (the user number to test every batch).",
    )

    parser.add_argument("--embed_dim", type=int, default=32, help="User / entity Embedding size.")
    parser.add_argument("--relation_dim", type=int, default=16, help="Relation Embedding size.")  # 64)

    parser.add_argument("--laplacian_type", type=str, default="random-walk", help="Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.",)
    parser.add_argument("--aggregation_type", type=str, default="bi-interaction", help="Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.",)
    parser.add_argument("--conv_dim_list", nargs="?", default="[32, 16]",  # [64, 32, 16] help="Output sizes of every aggregation layer.",
    )
    parser.add_argument("--mess_dropout", nargs="?", default="[0.1, 0.1]",  # [0.1, 0.1, 0.1] help="Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.",
    )

    parser.add_argument("--kg_l2loss_lambda", type=float, default=1e-5, help="Lambda when calculating KG l2 loss.",)
    parser.add_argument("--cf_l2loss_lambda", type=float, default=1e-5, help="Lambda when calculating CF l2 loss.",)

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--n_epoch", type=int, default=100, help="Number of epoch."  # 1000
    )
    parser.add_argument("--stopping_steps", type=int, default=100, help="Number of epoch for early stopping",)

    parser.add_argument("--cf_print_every", type=int, default=100, help="Iter interval of printing CF loss.")
    parser.add_argument("--kg_print_every", type=int, default=100, help="Iter interval of printing KG loss.")
    parser.add_argument("--evaluate_every", type=int, default=1, help="Epoch interval of evaluating CF.",)

    parser.add_argument("--Ks", nargs="?", default="[20, 40, 60]",  help="Calculate metric@K when evaluating.")

    args = parser.parse_args()

    save_dir = "trained_model/KGAT/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/".format(
        args.data_name,
        args.embed_dim,
        args.relation_dim,
        args.laplacian_type,
        args.aggregation_type,
        "-".join([str(i) for i in eval(args.conv_dim_list)]),
        args.lr,
        args.use_pretrain,
    )
    os.makedirs(save_dir, exist_ok = True)
    args.save_dir = save_dir

    return args

args = parse_kgat_args(sys.argv[1:])


# setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load data

if args.loader_pickle == "none":
    print("constructing new data, if the args is wrong") 
    print("please turn off the terminal")

    with open("dataloader_500.pkl", "wb") as f:
        data = DataLoaderKGAT(args)
        pickle.dump(data, f)
        print("dumped pickle!")

else:
    with open(args.loader_pickle, "rb") as f:
        print("bringing the data loader from pickle")
        data = pickle.load(f)
        print("loaded pickle")


print("loading finished...")
if args.use_pretrain == 1:
    user_pre_embed = torch.tensor(data.user_pre_embed)
    item_pre_embed = torch.tensor(data.item_pre_embed)
else:
    user_pre_embed, item_pre_embed = None, None

# model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
model = KGAT(
    args,
    data.n_users,
    data.n_entities,
    data.n_relations,
    data.A_in,
    user_pre_embed,
    item_pre_embed,
) 
if args.use_pretrain == 2:
    model = load_model(model, args.pretrain_model_path)

model.to(device)

# cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
cf_optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
# kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)
kg_optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

# initialize metrics
best_epoch = -1
best_recall = 0

Ks = eval(args.Ks)
k_min = min(Ks)
k_max = max(Ks)

epoch_list = []
metrics_list = {k: {"precision": [], "recall": [], "ndcg": []} for k in Ks}


for epoch in range(1, args.n_epoch + 1):
    time0 = time()
    model.train()

    # train cf
    time1 = time()
    cf_total_loss = 0
    n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

    for iter in range(1, n_cf_batch + 1):
        time2 = time()
        cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(
            data.train_user_dict, data.cf_batch_size
        )
        # cf_batch_user = cf_batch_user.to(device)
        # cf_batch_pos_item = cf_batch_pos_item.to(device)
        # cf_batch_neg_item = cf_batch_neg_item.to(device)
        cf_batch_user = cf_batch_user.to(device)
        cf_batch_pos_item = cf_batch_pos_item.to(device)
        cf_batch_neg_item = cf_batch_neg_item.to(device)

        
        cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode="train_cf")
        

        if np.isnan(cf_batch_loss.cpu().detach().numpy()):
            print('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(
                epoch, iter, n_cf_batch
                )
            )
            
        cf_batch_loss.backward()
        cf_optimizer.step()
        cf_optimizer.zero_grad()
        cf_total_loss += cf_batch_loss.item()

        if (iter % args.cf_print_every) == 0:
            print(
                "CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}".format(
                    epoch,
                    iter,
                    n_cf_batch,
                    time() - time2,
                    cf_batch_loss.item(),
                    cf_total_loss / iter,
                )
            )

        # break

    # train kg
    time3 = time()
    kg_total_loss = 0
    n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

    for iter in range(1, n_kg_batch + 1):
        time4 = time()
        (
            kg_batch_head,
            kg_batch_relation,
            kg_batch_pos_tail,
            kg_batch_neg_tail,
        ) = data.generate_kg_batch(
            data.train_kg_dict, data.kg_batch_size, data.n_users_entities
        )
 
        kg_batch_head = kg_batch_head.to(device)
        kg_batch_relation = kg_batch_relation.to(device)
        kg_batch_pos_tail = kg_batch_pos_tail.to(device)
        kg_batch_neg_tail = kg_batch_neg_tail.to(device)

        
        kg_batch_loss = model(
            kg_batch_head,
            kg_batch_relation,
            kg_batch_pos_tail,
            kg_batch_neg_tail,
            mode="train_kg",
        )

        if np.isnan(kg_batch_loss.cpu().detach().numpy()):
            print(
                "ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.".format(
                    epoch, iter, n_kg_batch
                )
            )

        kg_batch_loss.backward()
        kg_optimizer.step()
        kg_optimizer.zero_grad()
        kg_total_loss += kg_batch_loss.item()

        if (iter % args.kg_print_every) == 0:
            print(
                "KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}".format(
                    epoch,
                    iter,
                    n_kg_batch,
                    time() - time4,
                    kg_batch_loss.item(),
                    kg_total_loss / iter,
                )
            )

        # break
    print(
        "KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}".format(
            epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch
        )
    )

    # update attention
    time5 = time()
    h_list = data.h_list.to(device)
    t_list = data.t_list.to(device)
    r_list = data.r_list.to(device)
    relations = list(data.laplacian_dict.keys())
    model(h_list, t_list, r_list, relations, mode="update_att")
    print(
        "Update Attention: Epoch {:04d} | Total Time {:.1f}s".format(
            epoch, time() - time5
        )
    )

    print(
        "CF + KG Training: Epoch {:04d} | Total Time {:.1f}s".format(
            epoch, time() - time0
        )
    )

    # evaluate cf
    if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
        time6 = time()
        metrics_dict = evaluate(model, data, Ks, device)
        print(
            "CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
                epoch,
                time() - time6,
                metrics_dict[k_min]["precision"],
                metrics_dict[k_max]["precision"],
                metrics_dict[k_min]["recall"],
                metrics_dict[k_max]["recall"],
                metrics_dict[k_min]["ndcg"],
                metrics_dict[k_max]["ndcg"],
            )
        )

        epoch_list.append(epoch)
        for k in Ks:
            for m in ["precision", "recall", "ndcg"]:
                metrics_list[k][m].append(metrics_dict[k][m])
        best_recall, should_stop = early_stopping(
            metrics_list[k_min]["recall"], args.stopping_steps
        )

        if should_stop:
            break
        
       
        if metrics_list[k_min]["recall"].index(best_recall) == len(epoch_list) - 1:
            save_model(model, args.save_dir, epoch, best_epoch)
            print("Save model on epoch {:04d}!".format(epoch))
            best_epoch = epoch

    if epoch % 1 == 0:
        # save metrics
        metrics_df = [epoch_list]
        metrics_cols = ["epoch_idx"]
        for k in Ks:
            for m in ["precision", "recall", "ndcg"]:
                metrics_df.append(metrics_list[k][m])
                metrics_cols.append("{}@{}".format(m, k))
        metrics_df = pd.DataFrame(metrics_df).transpose()
        metrics_df.columns = metrics_cols
        metrics_df.to_csv(args.save_dir + "/metrics.tsv", sep="\t", index=False)

        # print best metrics
        best_metrics = (
            metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
        )
        print(
            "Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
                int(best_metrics["epoch_idx"]),
                best_metrics["precision@{}".format(k_min)],
                best_metrics["precision@{}".format(k_max)],
                best_metrics["recall@{}".format(k_min)],
                best_metrics["recall@{}".format(k_max)],
                best_metrics["ndcg@{}".format(k_min)],
                best_metrics["ndcg@{}".format(k_max)],
            )
        )


# save metrics
metrics_df = [epoch_list]
metrics_cols = ["epoch_idx"]
for k in Ks:
    for m in ["precision", "recall", "ndcg"]:
        metrics_df.append(metrics_list[k][m])
        metrics_cols.append("{}@{}".format(m, k))
metrics_df = pd.DataFrame(metrics_df).transpose()
metrics_df.columns = metrics_cols
metrics_df.to_csv(args.save_dir + "/metrics.tsv", sep="\t", index=False)

# print best metrics
best_metrics = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
print(
    "Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
        int(best_metrics["epoch_idx"]),
        best_metrics["precision@{}".format(k_min)],
        best_metrics["precision@{}".format(k_max)],
        best_metrics["recall@{}".format(k_min)],
        best_metrics["recall@{}".format(k_max)],
        best_metrics["ndcg@{}".format(k_min)],
        best_metrics["ndcg@{}".format(k_max)],
    )
)
