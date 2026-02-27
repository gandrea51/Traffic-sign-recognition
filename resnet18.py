import torch
import torch.nn as nn, torch.optim as optim, numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from dataset import dataloaders
from models.resnet import ResNet18GTSRB
from utils.trainer import training, evaluating, set_weights
from utils.early_stopping import EarlyStopping
from utils.visual import set_confusion, get_confusion

EPOCHS = 30
FREEZE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = 43

def main():
    train_loader, val_loader, test_loader = dataloaders(root='./GTSRB/Training')
    
    model = ResNet18GTSRB(CLASSES, pretrained=True).to(DEVICE)
    model.freeze()

    weights = set_weights(train_loader.dataset.base).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stop = EarlyStopping(patience=7, path='./networks/resnet18.pth')

    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch + 1} / {EPOCHS}')

        if epoch == FREEZE:
            print('\nUnfreezing entire network\n')
            model.unfreeze()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
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
    model.load_state_dict(torch.load('./networks/resnet18.pth'))
    
    test_loss, test_acc, test_prec, preds, labels = evaluating(model, test_loader, DEVICE)
    print(f'Test loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f}')

    cm = set_confusion(preds, labels)
    get_confusion(cm, path='./networks/resnet18.png')

if __name__ == '__main__':
    main()
