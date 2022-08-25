import torch

from torch.distributions import Categorical

class Generator:
    
    def __init__(self, model, tokenizer, vocab, device="cuda"):
        
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
    
    def generate(self, prompt, steps, stochastic=False):

        idxs = self.vocab(self.tokenizer(prompt))
        idxs = torch.tensor(idxs, dtype=torch.int64).unsqueeze(0)
        out_idxs = self.generate_loop(idxs, steps, stochastic)
        
        itos = self.vocab.get_itos()
        out_tokens = list(map(lambda idx: itos[idx], out_idxs))
        return out_tokens

    @torch.no_grad()
    def generate_loop(self, ctx, steps, stochastic):

        out = ctx.to(self.device)
        for _ in range(steps):
            logits = self.model(out).squeeze()
            next_logits = logits[-1, :]

            if stochastic:
                next_dist = Categorical(logits=next_logits)
                next_idx = next_dist.sample().reshape(1, 1)
            else:
                next_idx = torch.argmax(next_logits, dim=-1).reshape(1, 1)

            out = torch.cat((out, next_idx), dim=-1)

        return out.squeeze().tolist()