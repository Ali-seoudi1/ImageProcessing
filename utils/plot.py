from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history, output_dir):
    output_dir = Path(output_dir)

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row_index in range(confusion_matrix.shape[0]):
        for column_index in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row_index, column_index]
            plt.text(
                column_index,
                row_index,
                f"{value:d}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_domain_examples(dataset, output_path, num_examples=6):
    output_path = Path(output_path)
    examples_to_show = min(num_examples, len(dataset))

    plt.figure(figsize=(12, 6))
    for index in range(examples_to_show):
        image, label = dataset.get_raw_image(index)
        plt.subplot(2, 3, index + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
