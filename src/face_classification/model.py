from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet34_Weights
class PretrainedResNet34(nn.Module):
    def __init__(self, fine_tuning: bool = True, num_classes: int = 2):
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

    def forward(self, x):
        return self.model(x)