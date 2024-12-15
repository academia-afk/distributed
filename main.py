import torch
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(config):
    import torch
    import torchvision.models as models
    import torch.optim as optim

    # Example: Simple ResNet training setup
    model = models.resnet18()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # Dummy training loop
    for epoch in range(5):
        optimizer.zero_grad()
        # Simulate a batch of data
        data = torch.randn(32, 3, 224, 224)
        labels = torch.randint(0, 1000, (32,))
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Create a Trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    train_loop_config={"lr": 0.01},
)

# Execute Training
results = trainer.fit()
print(results)
