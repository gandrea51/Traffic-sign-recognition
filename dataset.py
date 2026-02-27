import os, cv2, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

'''
SIZE = 64
BATCH_SIZE = 64, 128 per un maggiore throughput
WORKERS = 6, In genere, workers circa pari alla metà del num. thread disponibili
SEED = 57
'''

class GTSRBDataset(Dataset):
    def __init__(self, root='./GTSRB/Training'):    
        self.root = root
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        '''
        Lettura delle directory, Apertura del .CSV e salvataggio dei metadata
        '''
        for folder in os.listdir(self.root):
            class_path = os.path.join(self.root, folder)
            if not os.path.isdir(class_path):
                continue
                
            csv = os.path.join(class_path, f'GT-{folder}.csv')
            if not os.path.exists(csv):
                continue

            f = pd.read_csv(csv, sep=';')
            for _, row in f.iterrows():
                image = os.path.join(class_path, row['Filename'])
                self.samples.append({
                    'image': image,
                    'label': int(row['ClassId']),
                    'roi': (
                        int(row['Roi.X1']),
                        int(row['Roi.Y1']),
                        int(row['Roi.X2']),
                        int(row['Roi.Y2'])
                    )
                })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.cvtColor(cv2.imread(sample['image']), cv2.COLOR_BGR2RGB)
        
        x1, y1, x2, y2 = sample['roi']
        image = image[y1:y2, x1:x2]
        
        label = sample['label']
        return image, label

class GTSRBSubset(Dataset):
    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = indices
        self.trasform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]

        if self.trasform:
            image = self.trasform(image)

        return image, torch.tensor(label, dtype=torch.long)

def operations(img_size=64):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),        
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def dataloaders(root, batch=64, img_size=64, num_workers=8, seed=57):
    original = GTSRBDataset()

    labels = [sample['label'] for sample in original.samples]
    indices = np.arange(len(labels))

    train_index, temp_index, y_train, y_temp = train_test_split(indices, labels, test_size=0.30, stratify=labels, random_state=seed)
    val_index, test_index = train_test_split(temp_index, test_size=0.50, stratify=y_temp, random_state=seed)

    train_tf, val_tf = operations()

    train_ds = GTSRBSubset(original, train_index, train_tf)
    val_ds = GTSRBSubset(original, val_index, val_tf)
    test_ds = GTSRBSubset(original, test_index, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def usage():
    train_loader, val_loader, test_loader = dataloaders(root='./GTSRB/Training')
    print('Batch di train:', len(train_loader))
    print('Batch di validation:', len(val_loader))
    print('Batch di test:', len(test_loader))

if __name__ == '__main__':
    usage()