import os
import tarfile
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image
import wandb
import ray
import glob


# Define a custom dataset for Stanford Dogs
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []

        # Load annotations
        annotation_file = os.path.join(root_dir, f"{split}_annotations.txt")
        with open(annotation_file, "r") as f:
            for line in f:
                image_path, label = line.strip().split(",")
                self.data.append((image_path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(os.path.join(self.root_dir, image_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Function to download and prepare the Stanford Dogs dataset
def prepare_stanford_dogs_dataset(root_dir="./data/stanford_dogs"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"

    # Download dataset
    for url in [dataset_url, annotations_url]:
        filename = url.split("/")[-1]
        filepath = os.path.join(root_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{filename} downloaded.")

        # Extract tar files
        print(f"Extracting {filename}...")
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(path=root_dir)

    # Process annotations to create split files
    process_annotations(root_dir)


def process_annotations(root_dir):
    annotations_dir = os.path.join(root_dir, "Annotation")
    images_dir = os.path.join(root_dir, "Images")
    train_output_file = os.path.join(root_dir, "train_annotations.txt")
    test_output_file = os.path.join(root_dir, "test_annotations.txt")

    # Create train and test annotation files
    with open(train_output_file, "w") as train_file, open(test_output_file, "w") as test_file:
        # Iterate through all class folders in the Annotation directory
        for class_id, class_dir in enumerate(sorted(glob.glob(os.path.join(annotations_dir, "*")))):
            class_name = os.path.basename(class_dir)

            # Parse all XML files in this class's annotation folder
            for xml_file in glob.glob(os.path.join(class_dir, "*.xml")):
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image filename
                image_filename = root.find("filename").text
                image_relative_path = os.path.join("Images", class_name, image_filename)

                # Split into train and test (80/20 split for simplicity)
                if "test" in image_filename.lower():
                    test_file.write(f"{image_relative_path},{class_id}\n")
                else:
                    train_file.write(f"{image_relative_path},{class_id}\n")

    print("Annotation files processed successfully.")

# Training function for each node
@ray.remote(num_gpus=1)
def train_on_node(node_id, config):
    wandb.init(
        project="ray-wandb-dogs",
        group="distributed-nodes",
        name=f"node_{node_id}",
        config=config,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset = StanfordDogsDataset(
        root_dir="./data/stanford_dogs", split="train", transform=transform["train"]
    )
    val_dataset = StanfordDogsDataset(
        root_dir="./data/stanford_dogs", split="test", transform=transform["val"]
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=config["num_nodes"], rank=node_id)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 120)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "accuracy": accuracy,
        })

    wandb.finish()
    return f"Node {node_id} finished training."


if __name__ == "__main__":
    ray.init(address="auto")

    # Prepare dataset
    prepare_stanford_dogs_dataset()

    # Training config
    config = {
        "lr": 0.001,
        "batch_size": 32,
        "num_epochs": 10,
        "num_nodes": 2,
    }

    futures = [train_on_node.remote(node_id, config) for node_id in range(config["num_nodes"])]
    results = ray.get(futures)
    print(results)

    ray.shutdown()
