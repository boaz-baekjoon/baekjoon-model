import os
import sys
import time
import argparse
import torch

from sasrec.data_loader import *
from sasrec.model import SASRec
from utils.evaluate import *


def readArguments(opts=sys.argv[1:]): # run baekjoon-model folder
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dir", required=True, help="path where training information will be saved")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=50, type=int, help="maximum length of past user history")
    parser.add_argument("--hidden_units", default=50, type=int, help="dim of item features")
    parser.add_argument("--num_blocks", default=2, type=int, help="num of attention and ffn layer sets")
    parser.add_argument("--num_epochs", default=100,type=int)
    parser.add_argument("--num_heads", default=1, type=int, help="num of attention layer heads")
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default='cpu', type=str) # mps
    parser.add_argument("--inference_only", default=False, type=bool)
    parser.add_argument("--state_dict_path", default=None, type=str)

    args = parser.parse_args(opts)
    return args

args = readArguments(sys.argv[1:])

if not os.path.isdir('./model/results/sasrec/' + args.dataset + '_' + args.train_dir):
    os.makedirs('./model/results/sasrec/' + args.dataset + '_' + args.train_dir, exist_ok=False)
with open(os.path.join('./model/results/sasrec/' + args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# load dataset
dataset = data_partition(args.dataset)
user_train, user_valid, user_test, usernum, itemnum = dataset
num_batch = len(user_train) // args.batch_size

# cc = 0.0
# for u in user_train:
#     cc += len(user_train[u])
# print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join('./model/results/sasrec/' + args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

model = SASRec(usernum, itemnum, args).to(args.device)

for name, param in model.named_parameters(): # initialize param
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass 

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)
        # print('pdb enabled for your quick check, pls type exit() if you do not need it')
        # import pdb; pdb.set_trace()
        
        
# training
bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # just to decrease identition
    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sample_batch(user_train, usernum, itemnum, args.batch_size, args.maxlen, 2023)
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
        # if step % 50 == 0 :
        #     print("\neye ball check raw_logits:"); print(pos_logits.sum()); print(neg_logits.sum()) # check pos_logits > 0, neg_logits < 0
        
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        
        # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

    if epoch % 20 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end='')
        t_test = evaluate(model, dataset, args)
        t_valid = evaluate_valid(model, dataset, args)
        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        f.write(str(t_valid) + ' ' + str(t_test) + '\n')
        f.flush()
        t0 = time.time()
        model.train()

    if epoch == args.num_epochs:
        folder = './model/results/sasrec/' + args.dataset + '_' + args.train_dir
        fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()
print("Done")
