import os

import matplotlib.pyplot as plt

figures_path = os.path.join("reports", "figures")


def plot_train_loss(statistics):
    _plot(statistics["train_loss"], "Train", "Loss", figures_path)


def plot_train_acc(statistics):
    _plot(statistics["train_acc"], "Train", "Accuracy", figures_path)


def plot_val_loss(statistics):
    _plot(statistics["val_loss"], "Validation", "Loss", figures_path)


def plot_val_acc(statistics):
    _plot(statistics["val_acc"], "Validation", "Accuracy", figures_path)


def _plot(y, train_type, metric_type, figures_path):
    plt.figure(figsize=(10, 5))
    x = range(1, len(y) + 1)
    plt.plot(x, y, label=f"{train_type} {metric_type}")
    plt.title(f"{train_type} {metric_type}")
    plt.legend()
    plt.savefig(os.path.join(figures_path, f"{train_type}_{metric_type}.png"))
