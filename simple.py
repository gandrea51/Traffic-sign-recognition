import torch
import torch.optim as optim
import numpy as np
from collections import Counter

from dataset import dataloaders
from models.simplecnn import SimpleCNN
from utils.trainer import training, evaluating
from utils.early_stopping import EarlyStopping
from utils.visual import set_confusion, get_confusion

EPOCHS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_loader, val_loader, test_loader = dataloaders(root='./GTSRB/Training')
    
    labels = [sample['label'] for sample in train_loader.dataset.base.samples]
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = np.array([counter[c] for c in classes])

    # Weights Log-smoothed
    k = 1.02
    weights = 1.0 / np.log(k + counts)
    weights = weights / weights.sum() * len(weights)
    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    model = SimpleCNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stop = EarlyStopping(patience=7, path='./networks/simplecnn.pth')

    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch + 1} / {EPOCHS}')
        
        train_loss, train_acc = training(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_prec, _, _ = evaluating(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f'Train loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
        print(f'Val loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f}')
        
        early_stop(val_loss, model)
        if early_stop.early:
            print('Early stopping triggered')
            break

    print('\nLoading best model...')
    model.load_state_dict(torch.load('./networks/simplecnn.pth'))
    
    test_loss, test_acc, test_prec, preds, labels = evaluating(model, test_loader, DEVICE)
    print(f'Test loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f}')

    cm = set_confusion(preds, labels)
    get_confusion(cm, path='./networks/simplecnn.png')

if __name__ == '__main__':
    main()