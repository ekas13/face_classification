import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from face_classification.data import FaceDataset

figures_path = os.path.join("reports", "figures")


def data_statistics(datadir: str = "data/processed") -> None:
    """Compute dataset statistics and visualize the data."""

    # Load training, validation, and test datasets
    train_dataset = FaceDataset(data_path=datadir, mode="train")
    val_dataset = FaceDataset(data_path=datadir, mode="val")
    test_dataset = FaceDataset(data_path=datadir, mode="test")

    # Print dataset details
    print(f"Train dataset: {train_dataset.data_path}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")

    print(f"Validation dataset: {val_dataset.data_path}")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")
    print("\n")

    print(f"Test dataset: {test_dataset.data_path}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # Label distribution for training, validation, and test datasets
    train_label_distribution = torch.bincount(torch.tensor([target for _, target in train_dataset]))
    val_label_distribution = torch.bincount(torch.tensor([target for _, target in val_dataset]))
    test_label_distribution = torch.bincount(torch.tensor([target for _, target in test_dataset]))

    # Plotting label distributions
    plt.bar(torch.arange(len(train_label_distribution)), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(figures_path, f"train_label_distribution.png"))
    plt.close()

    plt.bar(torch.arange(len(val_label_distribution)), val_label_distribution)
    plt.title("Validation label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(figures_path, f"val_label_distribution.png"))
    plt.close()

    plt.bar(torch.arange(len(test_label_distribution)), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(figures_path, f"test_label_distribution.png"))
    plt.close()


if __name__ == "__main__":
    typer.run(data_statistics)
