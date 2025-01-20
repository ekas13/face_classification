import os.path

import pytest
import torch
from torch.utils.data import Dataset

from src.face_classification.data import FaceDataset


@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_dataset_length():
    """Testing dataset lengths."""
    train_dataset = FaceDataset(mode="train")
    assert isinstance(train_dataset, Dataset)
    assert len(train_dataset) == 244

    val_dataset = FaceDataset(mode="val")
    assert isinstance(val_dataset, Dataset)
    assert len(val_dataset) == 32
    
    test_dataset = FaceDataset(mode="test")
    assert isinstance(test_dataset, Dataset)
    assert len(test_dataset) == 32

@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_data_shape():
    """Testing data shape."""
    train_dataset = FaceDataset(mode="train")
    image, target = train_dataset[0]
    assert image.shape == (3, 256, 256)
    assert target.shape == torch.Size([])

    val_dataset = FaceDataset(mode="val")
    image, target = val_dataset[0]
    assert image.shape == (3, 256, 256)
    assert target.shape == torch.Size([])

    test_dataset = FaceDataset(mode="test")
    image, target = test_dataset[0]
    assert image.shape == (3, 256, 256)
    assert target.shape == torch.Size([])

@pytest.mark.skipif(not os.path.exists("data/processed"), reason="Data files not found")
def test_dataset_loading():
    """Test the FaceDataset loading and DataLoader functionality."""
    train_set = FaceDataset(mode="train")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4)

    for batch in train_dataloader:
        images, labels = batch
        assert images.shape == (4, 3, 256, 256), f"Expected images of shape (4, 3, 256, 256), got {images.shape}"
        assert labels.shape == (4,), f"Expected labels of shape (4,), got {labels.shape}"
        break
