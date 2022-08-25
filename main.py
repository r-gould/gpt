import yaml
import torch
import torch.nn as nn

from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer

from src.gpt import GPT
from src.generator import Generator
from src.trainer import Trainer
from data.datahandler import DataHandler
from data.preprocess import preprocess
from utils import plot_stats

def main(dataloaders, gpt_params, lr, epochs, vocab_size,
        pad_idx, load_model=False, save_model=True, device="cuda"):

    model = GPT(**gpt_params, vocab_size=vocab_size, 
                pad_idx=pad_idx).to(device)

    if load_model:
        model.load_state_dict(torch.load("gpt/saved/gpt.pt"))

    loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optim = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.95))

    trainer = Trainer(dataloaders, loss, optim)
    train_losses, valid_losses = trainer.train(model, epochs, validate=True,
                                               save_model=save_model, 
                                               device=device)

    plot_stats(train_losses, valid_losses)
    return model

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = WikiText2("gpt/data/downloads/", split=("train", "valid", "test"))
    tokenizer = get_tokenizer("basic_english", language="en")
    model_name = "gpt1"
    
    with open(f"gpt/{model_name}.yaml", "r") as stream:
        gpt_params = yaml.safe_load(stream)

    batch_size = 16
    window_size = gpt_params.get("window_size")

    handler = DataHandler(datasets)
    dataloaders, vocab, counter = handler.load_data(
        preprocess, tokenizer, batch_size, window_size, 
        min_freq=5)
    
    lr = 1e-4
    epochs = 50
    vocab_size = len(vocab)
    pad_idx = vocab["<pad>"]

    model = main(dataloaders, gpt_params, 
                 lr, epochs, vocab_size, 
                 pad_idx, device=device)

    # Example

    generator = Generator(model, tokenizer, vocab, device=device)
    result = generator.generate(prompt="the meaning of life is", 
                                steps=30)
    print("Result:", result)