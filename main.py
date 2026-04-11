import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import CONFIG
from domains.base_domain import BaseDomain
from domains.fourier_domain import FourierDomain
from models.mobilenet import get_model
from utils.dataset import get_dataloaders
from utils.evaluate import evaluate
from utils.plot import plot_confusion_matrix, plot_curves, save_domain_examples
from utils.train import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["spatial", "fourier"], default="spatial")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_domain(domain_name):
    if domain_name == "fourier":
        return FourierDomain()
    return BaseDomain()


def compute_confusion_matrix(predictions, labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1
    return matrix


def main():
    args = parse_args()
    set_seed(CONFIG["seed"])
    os.environ.setdefault("TORCH_HOME", str(Path(".torch").resolve()))

    domain = get_domain(args.domain)
    output_dir = Path("outputs") / domain.name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Starting experiment | domain={domain.name} | device={device} | "
        f"epochs={CONFIG['epochs']} | batch_size={CONFIG['batch_size']} | "
        f"image_size={CONFIG['image_size']} | lr={CONFIG['learning_rate']}"
    )
    train_loader, val_loader, test_loader = get_dataloaders(domain)

    save_domain_examples(train_loader.dataset, output_dir / "domain_examples.png")

    model = get_model(CONFIG["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(CONFIG["epochs"]):
        print(f"Running epoch {epoch + 1}/{CONFIG['epochs']} for domain={domain.name}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']} | "
            f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}"
        )

    test_loss, test_acc, predictions, labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss={test_loss:.4f} | Test Accuracy={test_acc:.4f}")

    confusion_matrix = compute_confusion_matrix(predictions, labels, CONFIG["num_classes"])
    class_names = [str(index) for index in range(CONFIG["num_classes"])]

    plot_curves(history, output_dir)
    plot_confusion_matrix(confusion_matrix, class_names, output_dir / "confusion_matrix.png")

    results = {
        "domain": domain.name,
        "config": CONFIG,
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": confusion_matrix.tolist(),
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    torch.save(model.state_dict(), output_dir / "mobilenet_weights.pth")


if __name__ == "__main__":
    main()
