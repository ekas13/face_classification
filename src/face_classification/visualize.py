import os
import matplotlib.pyplot as plt

def plot_loss(statistics, figures_path):
    plt.figure(figsize=(10, 5))
    plt.plot(statistics["train_loss"], label="Train Loss")
    plt.plot(statistics["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join(figures_path, "loss.png"))

def plot_accuracy(statistics, figures_path):
    plt.figure(figsize=(10, 5))
    plt.plot(statistics["train_acc"], label="Train Accuracy")
    plt.plot(statistics["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(figures_path, "accuracy.png"))