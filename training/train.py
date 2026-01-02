# importing neccessary libaries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os

# configuration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# importing the dataset

# defining transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

# loading the cifair-10 training dataset

train_dataset = datasets.CIFAR10(
    root='./data',  
    train=True,     
    download=True,  
    transform=transform
)

# loading the cifair-10 test dataset
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,    
    download=True,
    transform=transform
)

def build_model():
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.last_channel, NUM_CLASSES
    )

    return model.to(DEVICE)
model = build_model()

# create dataloaders from the datasets
def get_dataloaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda")
    )

    return train_loader, test_loader

# training loop

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# evaluation loop

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# training logic

def main():
    train_loader, test_loader = get_dataloaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_acc = evaluate(model, test_loader)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, "mobilenet_cifar10_fp32.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

# entry point 
if __name__ == "__main__":
    main()

