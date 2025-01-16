from pytorch_lightning.callbacks import Callback
from .visualize import plot_train_acc, plot_train_loss, plot_val_acc, plot_val_loss


# This class implements the callbacks that Pytorch lightning uses to track the metrics of the model during training and validation
class MetricTracker(Callback):
    def __init__(self):
        self.collection = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = trainer.callback_metrics
        train_loss = elogs.get("train_loss")
        train_acc = elogs.get("train_acc")
        if train_loss is not None:
            self.collection["train_loss"].append(train_loss.item())
        if train_acc is not None:
            self.collection["train_acc"].append(train_acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        elogs = trainer.callback_metrics
        val_loss = elogs.get("val_loss")
        val_acc = elogs.get("val_acc")
        if val_loss is not None:
            self.collection["val_loss"].append(val_loss.item())
        if val_acc is not None:
            self.collection["val_acc"].append(val_acc.item())

    def on_train_end(self, trainer, pl_module):
        plot_train_loss(self.collection)
        plot_train_acc(self.collection)

    def on_validation_end(self, trainer, pl_module):
        plot_val_loss(self.collection)
        plot_val_acc(self.collection)
