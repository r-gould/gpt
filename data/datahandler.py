import torchtext

from collections import Counter
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset

class DataHandler:

    def __init__(self, datasets):

        self.datasets = datasets

    def load_data(self, preprocess, tokenizer, batch_size, window_size, min_freq):
        
        train, valid, test = self.datasets
        train_map = to_map_style_dataset(train)
        vocab, counter = self.build_vocab(train_map, tokenizer, min_freq)
        pad_idx = vocab["<pad>"]
        wrapper = lambda batch: preprocess(batch, tokenizer, vocab, window_size, pad_idx)
        
        dataloaders = (self.dataloader(train, batch_size, wrapper),
                       self.dataloader(valid, batch_size, wrapper),
                       self.dataloader(test, batch_size, wrapper))
        
        return dataloaders, vocab, counter

    @staticmethod
    def dataloader(dataset, batch_size, collate_fn):
        
        return DataLoader(to_map_style_dataset(dataset), batch_size, 
                          shuffle=True, collate_fn=collate_fn)

    def build_vocab(self, dataset_map, tokenizer, min_freq):

        tokens = map(tokenizer, dataset_map)
        counter = Counter()
        for token in tokens:
            counter.update(token)

        vocab = torchtext.vocab.vocab(counter, min_freq=min_freq, 
                                      specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab, counter