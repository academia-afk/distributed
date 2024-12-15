import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger.integration import WandbLoggerCallback
from ray import tune
import wandb

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.rand(size, 10)
        self.labels = (self.data.sum(dim=1) > 5).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Training Function
def train_model(config):
    wandb.init(project="ray-wandb-test", group="distributed-training", config=config)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=int(config["batch_size"]))

    model = SimpleNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        scheduler.step()

        tune.report(loss=avg_loss)
        wandb.log({"loss": avg_loss, "epoch": epoch})

    wandb.finish()

# Define Search Space
config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([16, 32, 64]),
    "epochs": 30,
}

# Scheduler for Ray Tune
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

# Run Ray Tune Experiment
tune.run(
    train_model,
    config=config,
    num_samples=10,
    resources_per_trial={"cpu": 2, "gpu": 1},
    scheduler=scheduler,
    callbacks=[
        WandbLoggerCallback(
            project="ray-wandb-test",
            log_config=True
        )
    ]
)
