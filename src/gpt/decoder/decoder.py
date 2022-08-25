import torch
import torch.nn as nn

from .layers import MultiHeadAttention, AddNorm, FeedForward

class Decoder(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, d_v, d_k, d_ff, dropout):

        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(num_heads, d_model, d_v, d_k, d_ff, dropout)
            for _ in range(num_layers)]
        )

    def forward(self, ctx_embeds, ctx_pad_mask):

        ctx_embeds = self.drop(ctx_embeds)

        out = ctx_embeds
        for layer in self.decoder_layers:
            out = layer(out, ctx_pad_mask)

        return out

class DecoderLayer(nn.Module):

    def __init__(self, num_heads, d_model, d_v, d_k, d_ff, dropout):
        
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, d_model, d_v, d_k, masked=True)
        self.add_norm_1 = AddNorm(d_model, dropout)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, decoder_in, ctx_pad_mask):

        attn = self.attention(decoder_in, decoder_in, decoder_in, ctx_pad_mask)
        attn = self.add_norm_1(attn, decoder_in)
        out = self.feed_forward(attn)
        return self.add_norm_2(out, attn)