import json
import os
import random

import gym
import numpy as np
import ray
import torch
import torchvision
import torchvision.transforms as T
import wandb
from pycocotools.cocoeval import COCOeval
from ray.tune import register_env
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class EnvClass(gym.Env):
    def __init__(self, seed):
        self.seed(seed)


class CocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = T.ToTensor()(img)

        coco_img_id = self.ids[idx]  # Real COCO image ID

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([coco_img_id], dtype=torch.int64)
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_coco(model, data_loader, device, dataset_dir):
    model.eval()
    coco_gt = data_loader.dataset.coco
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
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    pred_file = os.path.join(dataset_dir, "predictions.json")
    with open(pred_file, "w") as f:
        json.dump(results, f, indent=2)

    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]
    return ap, ap50


@ray.remote(num_gpus=1)
def train_loop_per_worker(node_id, config):
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        register_env("seed_env", lambda config: EnvClass(seed))

    set_seed(40 + node_id)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl", rank=node_id, world_size=config["num_nodes"])

    train_dir = config["train_dir"]
    val_dir = config["val_dir"]

    train_img_folder = os.path.join(train_dir, "data")
    train_ann_file = os.path.join(train_dir, "labels.json")
    val_img_folder = os.path.join(val_dir, "data")
    val_ann_file = os.path.join(val_dir, "labels.json")

    train_dataset = CocoDataset(train_img_folder, train_ann_file)
    val_dataset = CocoDataset(val_img_folder, val_ann_file)

    train_sampler = DistributedSampler(
        train_dataset,
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

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.rpn.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"] * config["num_nodes"],
        momentum=0.9,
        weight_decay=0.0005,
    )

    model = DDP(model, device_ids=[0])
    
    wandb.init(
        project="sync_distributed",
        group="five_nodes",
        name=f"node_{node_id}",
        config=config
    )

    for epoch in range(config["num_epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)

        total_train_loss = 0.0
        for images, targets in train_loader:
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
            accuracy = ap50 * 100.0  # A naive measure, not standard for detection

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "ap": ap,
                "ap50": ap50,
                "accuracy": accuracy,
            })
            print(
                f"[Rank {node_id}] Epoch [{epoch + 1}/{config['num_epochs']}]: "
                f"Train Loss={avg_train_loss:.4f}, AP={ap:.4f}, AP50={ap50:.4f}"
            )
        else:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

    wandb.finish()
    num_nodes = config["num_nodes"]
    torch.save(model.state_dict(), f"sync_model_{num_nodes}_nodes.pth")
    return f"Node {node_id} finished training."


if __name__ == "__main__":
    ray.init(address="auto")

    config = {
        "train_dir": "/workspace/dataset/training",
        "val_dir":   "/workspace/dataset/validation",
        "num_classes": 20,
        "num_nodes": 5,
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 0.005,
    }

    futures = [train_loop_per_worker.remote(node_id, config) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)
    print(results)
    ray.shutdown()
