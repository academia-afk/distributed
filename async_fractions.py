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


class EnvClass(gym.Env):
    def __init__(self, seed):
        self.seed(seed)


@ray.remote
class ParameterServer:
    def __init__(self, model_state_dict):
        self.global_weights = model_state_dict
        self.accumulated_gradients = None
        self.num_updates = 0

    def get_weights(self):
        return self.global_weights

    def update_gradients(self, gradients):
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradients
        else:
            for key in gradients:
                self.accumulated_gradients[key] += gradients[key]
        self.num_updates += 1

    def apply_updates(self):
        for key in self.global_weights:
            self.global_weights[key] -= self.accumulated_gradients[key] / self.num_updates
        self.accumulated_gradients = None
        self.num_updates = 0
        return self.global_weights


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
def train_loop_per_worker(node_id, config, ps):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        register_env("seed_env", lambda config: EnvClass(seed))

    set_seed(40 + node_id)

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

    global_weights = ray.get(ps.get_weights.remote())
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.load_state_dict(global_weights)
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

    wandb.init(
        project="distributed_test",
        group="two_nodes",
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
            accuracy = ap50 * 100.0

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
    torch.save(model.state_dict(), f"async_model_{num_nodes}_nodes.pth")
    return f"Node {node_id} finished training."


if __name__ == "__main__":
    ray.init(address="auto")
    register_env("my_seeded_env", lambda config: EnvClass(config))

    config = {
        "train_dir": "/workspace/datasets/openimages_coco/train",
        "val_dir":   "/workspace/datasets/openimages_coco/val",
        "num_classes": 4,
        "num_nodes": 2,
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 0.005,
    }

    initial_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    ps = ParameterServer.remote(initial_model.state_dict())

    futures = [train_loop_per_worker.remote(node_id, config, ps) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)
    print(results)
    ray.shutdown()
