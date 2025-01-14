import torch

def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.sum(y_pred == y_true).item() / len(y_true)