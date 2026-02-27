import os, numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataset import GTSRBDataset

DIRECTORY = './graph'

def get_label(dataset):
    return Counter([sample['label'] for sample in dataset.samples])

def plot_distr(counter, path):
    classes = sorted(counter.keys())
    counts = np.array([counter[c] for c in classes])

    total = counts.sum()
    percentages = counts / total * 100

    # Ordinamento per numerosità
    sorted_indices = np.argsort(counts)
    sorted_classes = np.array(classes)[sorted_indices]
    sorted_counts = counts[sorted_indices]
    sorted_percentages = percentages[sorted_indices]

    plt.figure(figsize=(16, 8))
    bars = plt.bar(sorted_classes, sorted_counts)

    # Classi sotto media
    mean_count = counts.mean()
    for i, bar in enumerate(bars):
        if sorted_counts[i] < mean_count:
            bar.set_alpha(0.5)
    
    plt.axhline(mean_count, linestyle='--')
    plt.xlabel('Classe')
    plt.ylabel('Numero immagini')
    plt.title('Distribuzione classi GTSRB (ordinata per numerosità)')
    plt.xticks(sorted_classes, rotation=90)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

    print(f'\nMedia immagini per classe: {mean_count:.2f}')
    print(f'Classe meno rappresentata: {sorted_classes[0]} ({sorted_counts[0]} immagini)')
    print(f'Classe più rappresentata: {sorted_classes[-1]} ({sorted_counts[-1]} immagini)')

def main():
    dataset = GTSRBDataset()
    print(f'\nTotale immagini: {len(dataset)}')

    distribution = get_label(dataset)
    print('\nDistribuzione classi:')
    for cls, count in sorted(distribution.items()):
        print(f'Classe: {cls}: {count} immagini')
    
    plot_distr(distribution, path=os.path.join(DIRECTORY, 'distribution.png'))

if __name__ == '__main__':
    main()