import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, random, collections

def load_dataset(data_dir):
    '''Function per l'upload del dataset'''

    X = np.load(os.path.join(data_dir, 'X_augmented.npy'))
    Y = np.load(os.path.join(data_dir, 'y_augmented.npy'))

    #X = np.load(os.path.join(data_dir, 'X.npy'))
    #Y = np.load(os.path.join(data_dir, 'Y.npy'))

    print(f'Dataset originale: {X.shape[0]} immagini')
    return X, Y

def random_img(X, Y, number=5):
    '''Function per mostrare un numero di immagini'''

    idx = random.sample(range(len(X)), number)
    plt.figure(figsize=(15, 5))
    for i, index in enumerate(idx):
        plt.subplot(1, number, i+1)
        plt.imshow(X[index])
        plt.title(f'Classe: {Y[index]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_by_class(X, Y, class_id, number=3):
    '''Function per mostrare immagini di una specifica classe'''

    plt.figure(figsize=(number * len(class_id), 3))
    for i, cid in enumerate(class_id):
        idx = [j for j in range(len(Y)) if Y[j] == cid]
        selected = random.sample(idx, min(number, len(idx)))
        for j, idx in enumerate(selected):
            plt.subplot(len(class_id), number, i * number + j + 1)
            plt.imshow(X[idx])
            plt.title(f'Classe: {class_id}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


data_dir = './dataset'
#data_dir = './list'

X, Y = load_dataset(data_dir)
random_img(X, Y)
random_img(X, Y)
#show_by_class(X, Y, class_id=[7, 27, 30, 41], number=3)