import os
import json
import torch
import torchvision
import wandb
import ray
from ray.util.sgd.torch import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.cocoeval import COCOeval

# ----------------------
# COCO Dataset Wrapper
# ----------------------
class CocoDataset(CocoDetection):
    def __getitem__(self, idx):
        """
        Overriding the default CocoDetection __getitem__ to return
        (image_tensor, target_dict).
        """
        img, annotations = super().__getitem__(idx)
        img = T.ToTensor()(img)

        # Real COCO image ID
        coco_img_id = self.ids[idx]

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([coco_img_id], dtype=torch.int64)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ----------------------
# Evaluate with COCOeval
# ----------------------
def evaluate_coco(model, data_loader, device, dataset_dir):
    """
    Runs COCO evaluation (mAP, AP50, etc.) on the entire validation set.
    Returns (ap, ap50) as floats.
    """
    model.eval()
    coco_gt = data_loader.dataset.coco  # ground truth

    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1)
                        ],
                        "score": float(score),
                    })

    # Save predictions to JSON
    pred_file = os.path.join(dataset_dir, "predictions.json")
    with open(pred_file, "w") as f:
        json.dump(results, f, indent=2)

    # COCO evaluation
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap = coco_eval.stats[0]   # mAP @ IoU=0.5:0.95
    ap50 = coco_eval.stats[1] # mAP @ IoU=0.5
    return ap, ap50

# ----------------------
# Single worker training
# ----------------------
@ray.remote(num_gpus=1)
def train_on_node(node_id, config):
    # Initialize wandb
    wandb.init(
        project="ray-wandb-object-detection",
        group="distributed-nodes",
        name=f"node_{node_id}",
        config=config
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Directories for COCO dataset
    train_dir = config["train_dir"]
    val_dir   = config["val_dir"]

    # Coco dataset paths
    train_img_folder = os.path.join(train_dir, "data")
    train_ann_file   = os.path.join(train_dir, "labels.json")
    val_img_folder   = os.path.join(val_dir, "data")
    val_ann_file     = os.path.join(val_dir, "labels.json")

    # Create datasets
    train_dataset = CocoDataset(train_img_folder, train_ann_file)
    val_dataset   = CocoDataset(val_img_folder,   val_ann_file)

    # DistributedSampler for multi-node training
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=config["num_nodes"],
        rank=node_id,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Build Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Replace the classification head for our (num_classes) detection classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        momentum=0.9,
        weight_decay=0.0005
    )

    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0.0

        # -------------
        # Training Loop
        # -------------
        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -------------
        # Validation
        # -------------
        ap, ap50 = evaluate_coco(model, val_loader, device, val_dir)

        # A naive "accuracy" metric, not standard for detection
        # We'll treat AP50 * 100 as a proxy "accuracy"
        accuracy = ap50 * 100.0

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "ap": ap,
            "ap50": ap50,
            "accuracy": accuracy,
        })

        print(f"Node {node_id} | Epoch [{epoch+1}/{config['num_epochs']}]: "
              f"Train Loss = {avg_train_loss:.4f}, AP = {ap:.4f}, AP50 = {ap50:.4f}, Acc(naive)={accuracy:.2f}")

    wandb.finish()
    return f"Node {node_id} finished training."

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # Initialize Ray (connect to existing cluster or run locally)
    ray.init(address="auto")  # or comment out if just local

    # Training Configuration
    config = {
        "train_dir": "datasets/openimages_coco/train",  # your paths
        "val_dir":   "datasets/openimages_coco/val",
        "num_classes": 2,  # for "Cat" + background
        "lr": 0.005,
        "batch_size": 4,
        "num_epochs": 10,
        "num_nodes": 2,  # or however many Ray workers you have
    }

    futures = [train_on_node.remote(node_id, config) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)  # Wait for all remote tasks
    print(results)

    ray.shutdown()
