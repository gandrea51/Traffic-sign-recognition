import torch
from sklearn.metrics import precision_score

def get_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).sum().item()

def get_precision(preds, labels):
    return precision_score(labels, preds, average='macro', zero_division=0)

