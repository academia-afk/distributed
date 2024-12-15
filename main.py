import ray
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def train_func(worker_id):
    wandb.init(project="ray-wandb-test", group="distributed-training", name=f"worker_{worker_id}")

    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(40):
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"epoch": epoch, "loss": loss.item()})

if __name__ == "__main__":
    ray.init(address='auto')
    num_workers = 2
    futures = [ray.remote(lambda id=i: train_func(id)).remote() for i in range(num_workers)]
    results = ray.get(futures)
    print("Training completed on all workers!")
    ray.shutdown()
