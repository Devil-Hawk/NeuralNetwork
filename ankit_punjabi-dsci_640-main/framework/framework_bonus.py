import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.features = self.create_features()
        self.classifier = self.create_classifier()

    def create_features(self):
        return nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(9),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(9, 18, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(18),
            nn.Conv2d(18, 18, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(18),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(18, 36, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(36),
            nn.Conv2d(36, 36, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(36),
            nn.MaxPool2d(2, 2)
        )

    def create_classifier(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(36 * 4 * 4, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def prepare_data_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4)
    return train_loader, val_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = prepare_data_loaders()

    model = CIFAR10_CNN().to(device)
    print("Total parameters:", count_parameters(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    evaluate_model(model, test_loader, criterion, device)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def evaluate_model(model, loader, criterion, device):
    test_loss, test_acc = evaluate(model, loader, criterion, device)
    print(f'Results on test dataset, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main()
