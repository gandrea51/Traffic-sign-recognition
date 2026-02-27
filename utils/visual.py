import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def set_confusion(all_preds, all_labels):
    return confusion_matrix(all_labels, all_preds)

def get_confusion(cm, path='./networks/simplecnn.png'):
    plt.figure(figsize=(10.8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    