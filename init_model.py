import argparse
from model.sasrec.model import SASRec
import torch
import json

arg_dir = './model/results/sasrec/baekjoon_test/args.txt'
weight_dir = './model/results/sasrec/baekjoon_test/SASRec.epoch=5.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth'

class make_args():
    def __init__(self):
        #self.batch_size = 128
        pass
        
args = make_args()
        
with open(arg_dir, 'r') as file:
    for line in file:
        key, value = line.strip().split(',')
        setattr(args, key, value)


args.batch_size = int(args.batch_size)
args.lr = float(args.lr)
args.maxlen = int(args.maxlen)
args.hidden_units = int(args.hidden_units)
args.num_blocks = int(args.num_blocks)
args.num_epochs = int(args.num_epochs)
args.num_heads = int(args.num_heads)
args.dropout_rate = float(args.dropout_rate)
args.l2_emb = float(args.l2_emb)
args.inference_only = bool(args.inference_only)
args.usernum = int(args.usernum)
args.itemnum = int(args.itemnum)


# load model
infer_model = SASRec(args.usernum, args.itemnum, args).to(args.device)

infer_model.load_state_dict(torch.load(weight_dir, map_location=torch.device(args.device)))