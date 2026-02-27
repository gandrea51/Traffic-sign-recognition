import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f'Early stopping counter: {self.counter} / {self.patience}')

            if self.counter >= self.patience:
                self.early = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
    