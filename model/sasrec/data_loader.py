import numpy as np
from collections import defaultdict


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    # assume user/item index starting from 1
    f = open('./data/preproc_data/%s.txt' % fname, 'r')
    #f = open('./data/preproc_data/test.txt')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3: # sequence가 3보다 작은 경우 무조건 train set으로 할당
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else: # valid : 마지막에서 2번째, test: 마지막에서 1번째
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
            
    return [user_train, user_valid, user_test, usernum, itemnum]

# user_train, user_valid, user_test, usernum, itemnum = data_partition("test")

def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_batch(user_train, usernum, itemnum, batch_size, maxlen, SEED):
    # batch_size x maxlen (default : 128 x 200)
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        
        seq = np.zeros([maxlen], dtype=np.int32) # input tensor
        pos = np.zeros([maxlen], dtype=np.int32) # positive
        neg = np.zeros([maxlen], dtype=np.int32) # negative
        nxt = user_train[user][-1]
        
        idx = maxlen - 1
        
        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # if nxt != 0: nxt == 0 인 경우 나오지 않을 듯 하여 일단 제거
            neg[idx] = random_neg(1, itemnum + 1, ts) 
            nxt = i
            idx -= 1
            if idx == -1: break
        
        return (user, seq, pos, neg)
    
    np.random.seed(SEED)
    user_ids = []
    seqs = []
    poss = []
    negs = []
    for _ in range(batch_size):
        one_sample = sample()
        user_ids.append(one_sample[0])
        seqs.append(one_sample[1])
        poss.append(one_sample[2])
        negs.append(one_sample[3])
            
    return user_ids, seqs, poss, negs # [128, 128, 128, 128]

# sample_batch(user_train, usernum, itemnum, 128, 200, 1)