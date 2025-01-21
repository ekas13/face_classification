import numpy as np
import onnxruntime
from fastapi import FastAPI
from face_classification.train import train
from face_classification.evaluate import evaluate
app = FastAPI()
from PIL import Image


@app.get("/predict_single_image")
def predict_single_image(image_path: str):
    """Predict using ONNX model."""
    # Load the ONNX model
    model = onnxruntime.InferenceSession("models/checkpoints/model-epoch=29-val_acc=0.94.ckpt")

    # Define the test dataset and dataloader
    import torchvision.transforms as transforms

    # Define the test dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0).numpy()
    output = model.run(None, image)

    return {"output": output[0].tolist()}


@app.get("/train_model")
def train_model():
    """Train a model on the Face Dataset."""
    train()
    return {"message": "Model training completed."}

def evaluate_model():
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    evaluate()
    return {"message": "Model evaluation completed."}