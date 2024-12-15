import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import ray


# Define the training function for each node
@ray.remote(num_gpus=1)  # Reserve 1 GPU per node
def train_on_node(node_id, config):
    # Initialize wandb for logging
    wandb.init(
        project="ray-wandb-dogs",
        group="distributed-nodes",
        name=f"node_{node_id}",
        config=config,
    )
    
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data transformations for the Stanford Dogs dataset
    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load Stanford Dogs dataset
    train_dataset = datasets.StanfordDogs(
        root="./data", split="train", download=True, transform=transform["train"]
    )
    val_dataset = datasets.StanfordDogs(
        root="./data", split="test", download=True, transform=transform["val"]
    )

    # Use DistributedSampler for training
    train_sampler = DistributedSampler(train_dataset, num_replicas=config["num_nodes"], rank=node_id)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Load the ResNet50 model (pretrained on ImageNet)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 120)  # Modify for 120 dog breeds
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        # Training step
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100.0 * correct / total

        # Log metrics to wandb
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

    # Configuration for training
    config = {
        "lr": 0.001,
        "batch_size": 32,
        "num_epochs": 10,
        "num_nodes": 2,  # Adjust based on your cluster size
    }

    # Launch training on each node
    futures = [train_on_node.remote(node_id, config) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)  # Wait for all nodes to finish
    print(results)

    # Shutdown Ray
    ray.shutdown()
