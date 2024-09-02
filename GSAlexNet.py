import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define the AlexNet Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Path to your dataset
data_dir = 'fashion_images'

# Load the dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check for GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("CUDA isn't available; exiting program")
    sys.exit()

# Initialize the model, loss function, and optimizer
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return train_loss / len(train_loader), 100. * correct / total

# Validation function
def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return val_loss / len(val_loader), 100. * correct / total

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return test_loss / len(test_loader), 100. * correct / total

# Grid search parameters
epochs = [10, 20]
batch_sizes = [32, 64, 128]
optimizers = {'adam': optim.Adam, 'sgd': optim.SGD}
best_val_acc = 0
best_params = {}
best_model = None
best_train_losses = []
best_val_losses = []

# Run grid search
for epoch in epochs:
    for batch_size in batch_sizes:
        for opt_name, opt_fn in optimizers.items():
            
            print(f'Testing the following parameters: Epoch - {epoch}, Batch Size - {batch_size}, Optimizer - {opt_name}')
            
            # Update dataloaders with new batch size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model = AlexNet().to(device)
            optimizer = opt_fn(model.parameters(), lr=0.001)
            
            train_losses = []
            val_losses = []
            
            for e in range(1, epoch + 1):
                train_loss, train_acc = train(model, device, train_loader, optimizer)
                val_loss, val_acc = validate(model, device, val_loader)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f'Epoch: {e}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'epoch': epoch, 'batch_size': batch_size, 'optimizer': opt_name}
                best_model = model
                best_train_losses = train_losses
                best_val_losses = val_losses

# Plotting the losses for the best hyperparameters
plt.figure(figsize=(10, 5))
plt.plot(range(1, best_params['epoch'] + 1), best_train_losses, label='Train Loss')
plt.plot(range(1, best_params['epoch'] + 1), best_val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f"Loss per Epoch for Best Model (Batch Size: {best_params['batch_size']}, Optimizer: {best_params['optimizer']})")
plt.legend()
plt.show()

# Evaluate the best model on the test set
test_loss, test_acc = test(best_model, device, test_loader)
print(f'Best Hyperparameters: {best_params}')
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')