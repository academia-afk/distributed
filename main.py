import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import wandb
import ray

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


# Function to train on a single node
@ray.remote(num_gpus=1)  # Reserve 1 GPU per node
def train_on_node(node_id, config):
    # Initialize wandb
    wandb.init(
        project="ray-wandb-cifar10",
        group="distributed-nodes",
        name=f"node_{node_id}",
        config=config,
    )

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Use DistributedSampler for even data distribution
    sampler = DistributedSampler(dataset, shuffle=True, num_replicas=config["num_nodes"], rank=node_id)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model, loss, and optimizer
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        # Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Log metrics
        accuracy = 100.0 * correct / total
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "accuracy": accuracy,
        })

    wandb.finish()
    return f"Node {node_id} finished training."


if __name__ == "__main__":
    # Initialize Ray
    ray.init(address="auto")

    # Training Configuration
    config = {
        "lr": 0.001,
        "batch_size": 64,
        "num_epochs": 10,
        "num_nodes": 2,  # Adjust based on available nodes
    }

    # Launch training on each node
    futures = [train_on_node.remote(node_id, config) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)  # Wait for all nodes to finish
    print(results)

    # Shutdown Ray
    ray.shutdown()
