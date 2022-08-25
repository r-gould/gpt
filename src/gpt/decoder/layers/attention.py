import torch
import numpy as np
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_v, d_k, masked):
        
        super().__init__()

        self.num_heads = num_heads

        self.linear_V = nn.Linear(d_model, num_heads * d_v)
        self.linear_K = nn.Linear(d_model, num_heads * d_k)
        self.linear_Q = nn.Linear(d_model, num_heads * d_k)

        self.attention = ScaledDotProductAttention(d_k, masked)

        self.output = nn.Linear(num_heads * d_v, d_model)

    def forward(self, Q, K, V, pad_mask):
        
        inp = (self.linear_Q(Q), self.linear_K(K), self.linear_V(V))
        inp = map(self.split, inp)

        attn = self.attention(*inp, pad_mask)
        out = self.concat(attn)
        return self.output(out)

    def split(self, batch):
        # batch of shape (batch_size, seq_len, num_heads*d)
        
        batch_size, seq_len, _ = batch.shape
        batch = batch.reshape(batch_size, seq_len, self.num_heads, -1)
        return batch.permute(0, 2, 1, 3)

    def concat(self, batch):
        # batch of shape (batch_size, num_heads, seq_len, d_v)

        batch_size, _, seq_len, _ = batch.shape
        batch = batch.permute(0, 2, 1, 3)
        return batch.reshape(batch_size, seq_len, -1)

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, attn_masked):
        
        super().__init__()

        self.d_k = d_k
        self.attn_masked = attn_masked

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, pad_mask):

        scores = Q @ K.transpose(-1, -2) / np.sqrt(self.d_k)
        scores = self.apply_mask(scores, pad_mask)

        return self.softmax(scores) @ V

    def apply_mask(self, scores, pad_mask):

        _, _, seq_len_out, seq_len_in = scores.shape
        mask = pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, seq_len_out, 1)

        if self.attn_masked:
            attn_mask = torch.triu(torch.ones(seq_len_out, seq_len_in), diagonal=1).unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.to(mask.device)
            mask = (mask + attn_mask) > 0
            
        return scores - 1e9 * mask