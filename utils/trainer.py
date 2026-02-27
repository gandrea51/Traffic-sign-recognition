import torch
import torch.nn as nn, numpy as np
from collections import Counter
from utils.metrics import get_accuracy, get_precision

def set_weights(dataset, k=1.02):
    labels = [sample['label'] for sample in dataset.samples]
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = np.array([counter[c] for c in classes])

    weights = 1.0 / np.log(k + counts)
    weights = weights / weights.sum() * len(weights)

    return torch.tensor(weights, dtype=torch.float)


def training(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_correct = 0
    total_sample = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += get_accuracy(outputs, labels)
        total_sample += labels.size(0)

    return total_loss / total_sample, total_correct / total_sample


def evaluating(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_sample = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += get_accuracy(outputs, labels)
            total_sample += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = get_precision(all_preds, all_labels)

    return ( total_loss / total_sample, total_correct / total_sample, precision, all_preds, all_labels )