import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import models
from torchvision.models import ResNet34_Weights
# from torchmetrics import Accuracy
import pytorch_lightning as pl

from utils import accuracy
# Hyperparameters
LR = 1e-3
OPTIMIZER = "adam"  # Options: "adam", "sgd"
OPTIMIZER_OPTIONS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}


class PretrainedResNet34(pl.LightningModule):
    """
    A PyTorch Lightning module for training and fine-tuning a pre-trained ResNet34 model.

    This module utilizes a pre-trained ResNet34 model from PyTorch's torchvision library,
    with options for fine-tuning and adapting the model for binary or multiclass classification tasks.

    Attributes:
        model (torchvision.models.ResNet): The pre-trained ResNet34 model.
        criterion (nn.Module): Loss function for the model (CrossEntropyLoss by default).
        acc (torchmetrics.Accuracy): Metric for calculating accuracy during training, validation, and testing.

    Args:
        num_classes (int): The number of classes for the classification task. For binary classification,
                           the final layer uses a sigmoid activation, while for multiclass classification,
                           it uses a linear layer with the number of specified classes.
        fine_tuning (bool, optional): If True, freezes the weights of the convolutional layers and updates
                                      only the fully connected layers. Defaults to True.
    """

    def __init__(self, num_classes: int, fine_tuning: bool = True):
        """
        Initializes the pre-trained ResNet34Model with specified arguments.
        Args:
            fine_tuning (bool): If True, freezes convolutional layer weights (fine-tuning mode).
            num_classes (int): Number of classes for the output layer.
        """
        super(PretrainedResNet34, self).__init__()

        self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)

        if fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features
        if num_classes == 2:
            self.model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
        else:
            self.model.fc = nn.Linear(num_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        # self.acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        """Forward pass."""

        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.criterion(y_pred, target)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy(y_pred, target), prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, target = batch
        y_pred = self(img)
        loss = self.criterion(y_pred, target)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', accuracy(y_pred, target), prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch):
        """Test step."""
        img, target = batch
        y_pred = self(img)
        loss = self.criterion(y_pred, target)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_acc", accuracy(y_pred, target), prog_bar=True, on_epoch=True, on_step=False)      
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = OPTIMIZER_OPTIONS[OPTIMIZER](self.parameters(), lr=LR)
        return optimizer
