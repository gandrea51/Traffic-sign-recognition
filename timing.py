import time, torch, torch.nn as nn, torch.optim as optim
from dataset import dataloaders
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIGS = [
    {"batch": 32, "workers": 2},
    {"batch": 64, "workers": 4},
    {"batch": 64, "workers": 8},
    {"batch": 128, "workers": 8},
]

def one_epoch(model, train_loader):
    model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    startime = time.time()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    endtime = time.time()
    return endtime - startime

class SmallCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    for c in CONFIGS:
        print(f'\nTesting: batch={c['batch']} workers={c['workers']}')
        train_loader, _, _ = dataloaders(root='./GTSRB/Training', batch=c['batch'], num_workers=c['workers'])

        model = SmallCNN()
        time_token = one_epoch(model, train_loader)
        print(f'Tempo epoca: {time_token:.2f} secondi.')
    
if __name__ == '__main__':
    main()