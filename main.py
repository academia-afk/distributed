import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Define a more realistic model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Training function for a single worker
def train_func(worker_id):
    wandb.init(
        project="ray-wandb-test",
        group="distributed-training",
        name=f"worker_{worker_id}",
        config={"worker_id": worker_id, "lr": 0.01}
    )

    # Define the model, optimizer, and loss function
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(40):
        inputs = torch.randn(64, 10)
        targets = torch.randn(64, 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to W&B
        wandb.log({"epoch": epoch, "loss": loss.item()})

    wandb.finish()
    return f"Worker {worker_id} completed training!"

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address="auto")

    # Number of workers
    num_workers = 4

    # Launch distributed training tasks
    futures = [ray.remote(lambda id=i: train_func(id)).remote() for i in range(num_workers)]
    results = ray.get(futures)

    print("Training completed on all workers!")
    ray.shutdown()
