import torch
import torch.nn as nn

class AddNorm(nn.Module):

    def __init__(self, d_model, dropout):
        
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, output, input):

        return self.norm(input + self.drop(output))