import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.cocoeval import COCOeval
import json

num_classes = 20

class CocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)

        img = T.ToTensor()(img)
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


def evaluate_coco(model, data_loader, device, dataset_dir):
    model.eval()
    coco_gt = data_loader.dataset.coco

    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # predictions in eval mode

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
    print(f"COCO Evaluation: AP={ap:.4f}, AP50={ap50:.4f}")


def main():
    train_dir = "coco/training"
    val_dir = "coco/validation"

    train_img_folder = os.path.join(train_dir, "data")
    train_ann_file = os.path.join(train_dir, "labels.json")

    val_img_folder = os.path.join(val_dir, "data")
    val_ann_file = os.path.join(val_dir, "labels.json")

    # Create dataset
    train_dataset = CocoDataset(train_img_folder, train_ann_file)
    val_dataset = CocoDataset(val_img_folder, val_ann_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor with a new one (for your custom number of classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the device (GPU or CPU)
    model.to(device)

    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze the classifier and bounding box predictor in the ROI heads
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}")
        evaluate_coco(model, val_loader, device, val_dir)


if __name__ == "__main__":
    main()
