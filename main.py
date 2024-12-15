# Training function for a single worker
def train_func(worker_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add device check
    wandb.init(
        project="ray-wandb-test",
        group="distributed-training",
        name=f"worker_{worker_id}",
        config={"worker_id": worker_id, "lr": 0.001, "epochs": 50},
    )
    train_loader, val_loader, test_loader = get_dataloader()
    model = CNNModel().to(device)  # Move model to the device
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(wandb.config["epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
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
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
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
