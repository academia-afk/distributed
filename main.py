import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Pre-download dataset
def download_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# Load CIFAR-10 data
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

# Training function for a single worker
def train_func(worker_id):
    wandb.init(
        project="ray-wandb-test",
        group="distributed-training",
        name=f"worker_{worker_id}",
        config={"worker_id": worker_id, "lr": 0.001, "epochs": 50},
    )
    train_loader, val_loader, test_loader = get_dataloader()
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(wandb.config["epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        wandb.log({"epoch": epoch, "train_loss": train_loss / len(train_loader),
                   "val_loss": val_loss / len(val_loader), "accuracy": accuracy})

    wandb.finish()
    return f"Worker {worker_id} completed training!"

if __name__ == "__main__":
    download_dataset()  # Pre-download dataset
    ray.init(address="auto")
    num_workers = 4
    futures = [ray.remote(lambda id=i: train_func(id)).remote() for i in range(num_workers)]
    results = ray.get(futures)
    print("Training completed on all workers!")
    ray.shutdown()
