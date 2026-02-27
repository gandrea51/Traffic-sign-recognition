import os, numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from dataset import GTSRBDataset

DIRECTORY = './graph'

def inverse_frequency(counts):
    return 1.0 / counts

def inverse_sqrt_frequency(counts):
    return 1.0 / np.sqrt(counts)

def log_smoothed(counts, k=1.02):
    return 1.0 / np.log(k + counts)

def effective_number(counts, beta=0.999):
    return (1 - beta) / (1 - np.power(beta, counts))

def balanced_softmax(counts):
    return 1.0 / counts  # used differently inside logits, but shown for comparison

def normalize(weights):
    return weights / weights.sum() * len(weights)

def get_counts(dataset):
    labels = [sample['label'] for sample in dataset.samples]
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = np.array([counter[c] for c in classes])
    return classes, counts

def main():
    dataset = GTSRBDataset()
    classes, counts = get_counts(dataset)

    print(f'\nImmagini totali: {counts.sum()} | Numero classi: {len(classes)}')

    weight_dict = {
        "Inverse": normalize(inverse_frequency(counts)),
        "InverseSqrt": normalize(inverse_sqrt_frequency(counts)),
        "LogSmoothed": normalize(log_smoothed(counts, k=1.02)),
        "EffectiveNumber": normalize(effective_number(counts, beta=0.999)),
        "BalancedSoftmax": normalize(balanced_softmax(counts)),
    }

    header = f'{'Class':>5} {'Count':>6} '
    for key in weight_dict:
        header += f"{key:>18}"
    print(header)
    print('-' * len(header))

    for i, cls in enumerate(classes):
        row = f'{cls:5d} {counts[i]:6d} '
        for key in weight_dict:
            row += f'{weight_dict[key][i]:18.4f}'
        print(row)

    plt.figure(figsize=(14, 7))
    for key in weight_dict:
        plt.plot(classes, weight_dict[key], label=key)

    plt.xlabel('Classe'); plt.ylabel('Peso normalizzato')
    plt.title('Confronto strategie di pesatura')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(DIRECTORY, 'weights.png')
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    main()