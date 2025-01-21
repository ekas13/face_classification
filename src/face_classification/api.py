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
    import onnxruntime
    from PIL import Image
    import torchvision.transforms as transforms

    # Load the ONNX model
    model = onnxruntime.InferenceSession("models/model_final.onnx")

    # Define the test dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    image = transform(image)
    image = image.unsqueeze(0).numpy()

    # Get the input name for the model
    input_name = model.get_inputs()[0].name

    # Perform inference
    output = model.run(None, {input_name: image})
    output_class = int(np.argmax(output))

    return {"output class": output_class}


@app.get("/train_model")
def train_model():
    """Train a model on the Face Dataset."""
    train()
    return {"message": "Model training completed."}

@app.get("/evaluate_model")
def evaluate_model():
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    evaluate()
    return {"message": "Model evaluation completed."}