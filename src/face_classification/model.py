from torchvision import models
import torch.nn as nn
import torch
from torchvision.models import ResNet34_Weights
import pytorch_lightning as pl
class PretrainedResNet34(pl.LightningModule):
    def __init__(self,  num_classes: int, fine_tuning: bool = True):
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
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        else:
            self.model.fc = nn.Linear(num_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        return self.criterion(y_pred, target)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)