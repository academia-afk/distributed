import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

# Distributed training function
def train_func(rank, world_size, epochs=50):
    # Initialize distributed process group
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # Set device for each process
    device = torch.device(f"cuda:{rank}")

    # Initialize W&B
    wandb.init(
        project="ray-wandb-test",
        group="distributed-training",
        name=f"worker_{rank}",
        config={"lr": 0.001, "epochs": epochs},
    )

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Split dataset for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = CNNModel().to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        wandb.log({"epoch": epoch, "train_loss": train_loss / len(train_loader), "accuracy": accuracy})

    wandb.finish()
    dist.destroy_process_group()
    return f"Worker {rank} completed training!"

if __name__ == "__main__":
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "172.17.0.2"  # Head node IP
    os.environ["MASTER_PORT"] = "12355"       # Arbitrary port for communication

    ray.init(address="auto")

    # Number of workers and GPUs
    world_size = 2

    # Launch distributed training
    ray.get([
        ray.remote(train_func).remote(rank, world_size) for rank in range(world_size)
    ])

    print("Distributed training completed!")
    ray.shutdown()
