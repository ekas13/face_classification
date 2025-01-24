import os
import shutil
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile

import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary, make_asgi_app

from face_classification.evaluate import evaluate
from face_classification.train import train


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}.")


MY_REGISTRY = CollectorRegistry()
error_counter = Counter("prediction_error", "Number of prediction errors", registry=MY_REGISTRY)
request_counter = Counter("prediction_requests", "Number of prediction requests", registry=MY_REGISTRY)
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, request_counter, runtime_counter
    # Load the ONNX model
    model_path = "models/model_final.onnx"
    if not os.path.exists(model_path):
        bucket_name = "face-classification-models"
        source_blob_name = "models/model_final.onnx"
        download_from_gcs(bucket_name, source_blob_name, model_path)

    model = onnxruntime.InferenceSession(model_path)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    yield

    # Clean up
    del model
    del transform


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app(registry=MY_REGISTRY))


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Backend here!"}


@app.get("/predict_single_image")
def predict_single_image(image_path: str):
    """Predict using ONNX model."""
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    image = transform(image)
    image = image.unsqueeze(0).numpy()  #type: ignore

    # Get the input name for the model
    input_name = model.get_inputs()[0].name

    # Perform inference
    output = model.run(None, {input_name: image})
    probabilities = np.exp(output) / np.sum(np.exp(output))
    prediction = int(np.argmax(output))
    print(f"Predicted class: Person {prediction}")

    return probabilities, prediction


# FastAPI endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    request_counter.inc()
    with request_latency.time():
        try:
            # Save the uploaded file to a temporary location
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            # Use the saved file for prediction
            probabilities, prediction = predict_single_image(temp_file_path)

            # Remove the temporary file after use
            os.remove(temp_file_path)

            return {"filename": file.filename, "prediction": prediction, "probabilities": probabilities.tolist()}
        except Exception as e:
            error_counter.inc()
            raise HTTPException(status_code=500, detail=str(e))


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
