import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
def generate_data():
    X, y = make_classification(
        n_samples=5000, n_features=10, n_informative=8, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

# Define a more realistic model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        return self.fc(x)

# Training function for a single worker
def train_func(worker_id):
    # Initialize W&B
    wandb.init(
        project="ray-wandb-test",
        group="distributed-training",
        name=f"worker_{worker_id}",
        config={"worker_id": worker_id, "lr": 0.001}
    )

    # Generate data
    X_train, y_train, X_test, y_test = generate_data()

    # Define the model, optimizer, and loss function
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Evaluate accuracy on test set
        model.eval()
        with torch.no_grad():
            predictions = (model(X_test).squeeze() > 0.5).float()
            accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

        # Log metrics to W&B
        wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})

    wandb.finish()
    return f"Worker {worker_id} completed training!"

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address="auto", log_to_driver=True)

    # Number of workers
    num_workers = 4

    # Launch distributed training tasks
    futures = [ray.remote(lambda id=i: train_func(id)).remote() for i in range(num_workers)]
    results = ray.get(futures)

    print("Training completed on all workers!")
    ray.shutdown()
