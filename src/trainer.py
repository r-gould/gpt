import torch
import numpy as np

from tqdm import tqdm

class Trainer:

    def __init__(self, dataloaders, loss, scheduler):

        self.dataloaders = dataloaders
        self.loss = loss
        self.scheduler = scheduler

    def train(self, model, epochs, validate=True, save_model=True, device="cuda"):

        train_dl, valid_dl, _ = self.dataloaders
        train_losses = []
        valid_losses = []

        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            epoch_losses = []
            for data in tqdm(train_dl):
                
                data = data.to(device)

                logits = model(data[:, :-1])
                _, _, vocab_size = logits.shape
                loss = self.loss(logits.reshape(-1, vocab_size), 
                                 data[:, 1:].flatten())

                self.scheduler.zero_grad()
                loss.backward()
                self.scheduler.step()
                
                epoch_losses.append(loss.item())

            if save_model:
                print("Saving model...")
                torch.save(model.state_dict(), f"saved/gpt_{epoch}.pt")
                print("Model saved")

            avg_loss = np.mean(epoch_losses[-50:])
            print("Avg. train loss:", avg_loss)
            train_losses.append(avg_loss)
            epoch_losses = []
            
            if validate:
                valid_loss = self.test(model, valid_dl, device=device)
                print("Avg. valid loss:", valid_loss)
                valid_losses.append(valid_loss)
        
        return train_losses, valid_losses

    @torch.no_grad()
    def test(self, model, test_dl, device="cuda"):

        valid_losses = []

        for data in test_dl:

            data = data.to(device)

            logits = model(data[:, :-1])
            _, _, vocab_size = logits.shape
            loss = self.loss(logits.reshape(-1, vocab_size), 
                             data[:, 1:].flatten())

            valid_losses.append(loss.item())

        return np.mean(valid_losses)
