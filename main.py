@ray.remote(num_gpus=1)
def train_func(config, node_id):
    # Initialize wandb for each worker
    wandb.init(
        project="ray-wandb-object-detection",
        group="distributed-nodes",
        name=f"node_{node_id}",
        config=config
    )

    # Dataset setup
    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    train_img_folder = os.path.join(train_dir, "data")
    train_ann_file = os.path.join(train_dir, "labels.json")
    val_img_folder = os.path.join(val_dir, "data")
    val_ann_file = os.path.join(val_dir, "labels.json")

    train_dataset = CocoDataset(train_img_folder, train_ann_file)
    val_dataset = CocoDataset(val_img_folder, val_ann_file)

    # DataLoader setup
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=config["num_nodes"],
        rank=node_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model setup
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)

        total_train_loss = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        if node_id == 0:
            ap, ap50 = evaluate_coco(model, val_loader, device, val_dir)
            accuracy = ap50 * 100.0

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "ap": ap,
                "ap50": ap50,
                "accuracy": accuracy,
            })
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}]: "
                  f"Train Loss={avg_train_loss:.4f}, AP={ap:.4f}, AP50={ap50:.4f}")
        else:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    wandb.finish()  # Make sure to call finish at the end of each worker's training

