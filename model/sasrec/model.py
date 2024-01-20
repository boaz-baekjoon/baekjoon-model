import torch
import numpy as np
import random

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        # input : (batchsize, channel_in, dim), output : (batchsize, channel_out, dim_out) => input and output should transpose(1,2)
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # because kernel_size is 1, dim_out = dim.
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(1, 2))))))
        outputs = outputs.transpose(1, 2) 
        outputs += inputs # residual connection
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        
        # set parameter
        self.user_num = user_num
        self.item_num = item_num
        self.num_blocks = args.num_blocks
        self.dev = args.device
        
        # embbeding layer
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # self-attention layer
        self.attention_layernorms = torch.nn.ModuleList() # for Q
        self.attention_layers = torch.nn.ModuleList()
        
        # P-W FNN layer 
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps = 1e-8)
        
        
        for _ in range(self.num_blocks):
            # Query
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            # self-attention layer
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate, 
                                                         batch_first = True) # if batch_first=True then shape of input and output is (batchsize, seq len, dim)
            self.attention_layers.append(new_attn_layer)           
            
            # P-W FNN layer
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def model_layers(self, input_seqs):
        # embedding
        seqs = self.item_emb(torch.LongTensor(input_seqs).to(self.dev))
        # seqs *= self.item_emb.embedding_dim ** 0.5 # seq와 position의 범위를 맞춰주기 위함인가?
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0],1]) # 실험1
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev)) # 실험1
        seqs = self.emb_dropout(seqs)
        
        # timeline_mask
        timeline_mask = (seqs == 0).to(dtype=torch.bool, device=self.dev)
        seqs *= ~timeline_mask
        
        # self-attention layer and P-W FNN layer
        tl = seqs.shape[1] # item sequence len
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(self.num_blocks):
            # self-attention layer
            Q = self.attention_layernorms[i](seqs) # normalize
            
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                                      attn_mask=attention_mask)
            
            seqs = Q + mha_outputs # residual connection
            
            # P-W FNN layer
            seqs = self.forward_layernorms[i](seqs) # normalize
            seqs = self.forward_layers[i](seqs)
            
            seqs *= ~timeline_mask # timeline_mask
        
        output_seqs = self.last_layernorm(seqs)
        
        return output_seqs
            
    def forward(self, user_id, seqs, pos, neg):
        
        output_seqs = self.model_layers(seqs)
        
        # predict layer
        pos_emb = self.item_emb(torch.LongTensor(pos).to(self.dev))
        neg_emb = self.item_emb(torch.LongTensor(neg).to(self.dev))
        
        pos_logits = (output_seqs * pos_emb).sum(dim=-1)
        neg_logits = (output_seqs * neg_emb).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids, seqs, item_indices):
        with torch.no_grad():
            output_seqs = self.model_layers(seqs)
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
            print(output_seqs.shape)
            print(item_embs.shape)
            candidate_logits = output_seqs.squeeze().matmul(item_embs.T).sum(dim=0)

        return candidate_logits

    def predict_for_user_sq(self, sequence:list, item_num, problem_list, args):

        # padding
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(sequence):
            seq[idx] = i
            idx -= 1
            if idx == -1: 
                break
            
        # retrieval
        item_idx = random.choices(problem_list, k=100)
        
        predict = self.predict(*[np.array(l) for l in [[1], [seq], item_idx]]).cpu().detach()
        
        predict = np.array(predict, dtype=int)
        output = [item_idx[idx] for idx in list(np.argsort(predict)[-item_num:])]
        return output