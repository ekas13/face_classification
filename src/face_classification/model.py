# from torchmetrics import Accuracy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig
from torchvision import models
from torchvision.models import ResNet34_Weights

pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

# 🏋️‍♀️ Weights & Biases
import wandb

wandb.login()
wandb_logger = pl.loggers.WandbLogger(project="face_classification")
# ⚡ 🤝 🏋️‍♀️
# Optimizer options
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
        cfg (DictConfig): Configuration object containing model hyperparameters.
            - `cfg.model.num_classes`: The number of classes for the classification task. For binary classification,
                                    the final layer uses a sigmoid activation, while for multiclass classification,
                                    it uses a linear layer with the number of specified classes.
            - `cfg.model.fine_tuning`: If True, freezes the weights of the convolutional layers and updates
                                      only the fully connected layers.
            - `cfg.model.optimizer`: The optimizer configuration, including the type ("adam" or "sgd")
                                      and the learning rate.

    """

    def __init__(self, cfg: DictConfig):
        """
        Initializes the pre-trained ResNet34Model with specified arguments.
        Args:
            cfg (DictConfig): Configuration object containing model hyperparameters.
        """
        super(PretrainedResNet34, self).__init__()

        self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)

        if cfg.model.fine_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features
        if cfg.model.num_classes == 2:
            self.model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
        else:
            self.model.fc = nn.Linear(num_features, cfg.model.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_cfg = cfg.model.optimizer
        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes, average="macro")
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes, average="macro")
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes, average="macro")

    def forward(self, x):
        """Forward pass."""

        return self.model(x)

    def loss(self, batch, mode):
        """Common step function."""
        img, target = batch
        logits = self(img)
        loss = self.criterion(logits, target)
        return logits, loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        xs, ys = batch
        logits, loss = self.loss(batch, "train")
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, ys)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.valididation_step_targets = []
        self.validation_step_logits = []
        self.validation_step_preds = []

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        xs, ys = batch
        logits, loss = self.loss(batch, "val")
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, ys)
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", self.valid_acc, on_epoch=True)
        self.validation_step_logits.append(logits)
        self.valididation_step_targets.append(ys)
        self.validation_step_preds.append(preds)
        return loss

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_logits

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"val/logits": wandb.Histogram(flattened_logits.to("cpu")), "global_step": self.global_step}
        )
        # Compute confusion matrix
        targets = torch.cat(self.valididation_step_targets).cpu()
        preds = torch.cat(self.validation_step_preds).cpu()
        # Log confusion matrix
        self.logger.experiment.log(
            {
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=targets.numpy(), preds=preds.numpy(), class_names=[str(i) for i in range(16)]
                )
            }
        )

    def test_step(self, batch, batch_idx):
        """Test step."""
        xs, ys = batch
        logits, loss = self.loss(batch, "test")
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, ys)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        return loss

    def on_test_epoch_end(self):  # args are defined as part of pl API
        dummy_input = torch.zeros((1, 3, 256, 256), device=self.device)
        model_filename = "models/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        artifact = wandb.Artifact(name="model.ckpt", type="model")
        artifact.add_file(model_filename)
        wandb.log_artifact(artifact)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = OPTIMIZER_OPTIONS[self.optimizer_cfg.type](self.parameters(), lr=self.optimizer_cfg.lr)
        return optimizer
