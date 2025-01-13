from pathlib import Path
import typer
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from torchvision.transforms import ToTensor

class FaceDataset(Dataset):
    """My custom dataset for loading face images and their targets."""

    def __init__(self, data_path: str="data/processed", transform=ToTensor(), mode: str="train") -> None:
        """Initialize the dataset with the preprocessed data path.

        Args:
            data_path (Path): Path to the folder containing preprocessed data.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
            mode (str): Dataset mode (train, val, test).
        """
        self.data_path = Path(os.path.join(data_path, mode))
        self.transform = transform
        self.image_paths = sorted(self.data_path.glob("*.jpg"))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image, target) where target is the class label.
        """
        # Get the image path
        image_path = self.image_paths[index]

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Extract the class label from the filename
        filename = image_path.name
        class_label = int(filename.split("_")[1]) - 1  # Extract the number after 'person_'

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Convert the target to a tensor
        target = torch.tensor(class_label, dtype=torch.long)

        return image, target


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data so we have a cleaner folder structure of our data, with class labels contained in image names.
    
    Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image, target) where target is the class label.
    """
    print("Preprocessing data...")

    # Define paths for train, test, and val folders
    train_output_folder = output_folder / "train"
    test_output_folder = output_folder / "test"
    val_output_folder = output_folder / "val"

    # Ensure output directories exist
    train_output_folder.mkdir(parents=True, exist_ok=True)
    test_output_folder.mkdir(parents=True, exist_ok=True)
    val_output_folder.mkdir(parents=True, exist_ok=True)

    # Process "Training" folder
    train_folder = raw_data_path / "training"
    for person_folder in train_folder.iterdir():
        if person_folder.is_dir():
            person_id = person_folder.name.replace("face", "person_")
            for i, image_path in enumerate(person_folder.glob("*.jpg"), start=1):
                new_name = f"{person_id}_face_{i}.jpg"
                with Image.open(image_path) as img:
                    resized_image = img.resize((256, 256))
                    resized_image.save(train_output_folder / new_name)

    # Process "Testing" folder and split into test and val folders
    test_folder = raw_data_path / "testing"
    for person_folder in test_folder.iterdir():
        if person_folder.is_dir():
            person_id = person_folder.name.replace("face", "person_")
            image_paths = list(person_folder.glob("*.jpg"))
            num_images = len(image_paths)
            half_point = num_images // 2

            # Split images into val and test sets
            val_images = image_paths[:half_point]
            test_images = image_paths[half_point:]

            # Copy and resize val images
            for i, image_path in enumerate(val_images, start=1):
                new_name = f"{person_id}_face_{i}.jpg"
                with Image.open(image_path) as img:
                    resized_image = img.resize((256, 256))
                    resized_image.save(val_output_folder / new_name)

            # Copy and resize test images
            for i, image_path in enumerate(test_images, start=1):
                new_name = f"{person_id}_face_{i + half_point}.jpg"
                with Image.open(image_path) as img:
                    resized_image = img.resize((256, 256))
                    resized_image.save(test_output_folder / new_name)

    print(f"Data preprocessing complete. Preprocessed data saved to {output_folder}")

if __name__ == "__main__":
    typer.run(preprocess)
