import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Training function to be executed on each worker
def train_func():
    # Initialize Weights & Biases inside the worker function
    wandb.init(project="ray-wandb-test", group="distributed-training")

    # Initialize the model
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Dummy data for training
    for epoch in range(5):  # Number of epochs
        inputs = torch.randn(32, 10)  # Batch size of 32, input size of 10
        targets = torch.randn(32, 1)   # Corresponding targets

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to W&B
        wandb.log({"epoch": epoch, "loss": loss.item()})

# Main execution block to set up Ray and run training
if __name__ == "__main__":
    ray.init(address='auto')  # Connect to the existing Ray cluster

    # Define the number of workers (2 for two nodes)
    num_workers = 2

    # Launch distributed training on multiple workers
    futures = [ray.remote(train_func).remote() for _ in range(num_workers)]

    # Wait for all workers to finish training
    results = ray.get(futures)

    print("Training completed on all workers!")

    # Shutdown Ray after training is complete
    ray.shutdown()
