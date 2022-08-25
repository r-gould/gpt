import torch
import torch.nn as nn

from .decoder import Decoder

class GPT(nn.Module):

    def __init__(self, layers, num_heads, d_model, d_v, d_k, d_ff, 
                 dropout, window_size, vocab_size, pad_idx):

        super().__init__()

        self.window_size = window_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(window_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.decoder = Decoder(layers, num_heads, d_model, d_v, d_k, d_ff, dropout)
        
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, ctx):

        _, ctx_len = ctx.shape
        assert ctx_len <= self.window_size

        ctx_pad_mask = (ctx == self.pad_idx)

        ctx_embeds = self.embedding(ctx)
        ctx_embeds += self.pos_embedding.weight[:ctx_len]
        ctx_embeds = self.drop(ctx_embeds)

        decoded = self.decoder(ctx_embeds, ctx_pad_mask)
        return self.output(decoded)