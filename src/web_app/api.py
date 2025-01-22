import json
from contextlib import asynccontextmanager
import hydra
import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms
import numpy as np
from face_classification.model import PretrainedResNet34

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, imagenet_classes
    # Load model

    with hydra.initialize(config_path="../../configs", version_base=None, job_name="evaluate_model"):
        cfg = hydra.compose(config_name="default_config")

    # Load the model checkpoint
    model_checkpoint = torch.load("C:/Users/vbran/Desktop/DTU code/Semestar1/face_classification/models/checkpoints/model-epoch=29-val_acc=0.94.ckpt", map_location=torch.device('cpu'))
    model = PretrainedResNet34(cfg)  # Assuming the model architecture is ResNet18
    model.load_state_dict(model_checkpoint['state_dict'])
    model.eval()

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


def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    output_class = str(int(np.argmax(output, 1)))
    return output_class


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint for image classification
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        async with await anyio.open_file(file.filename, "wb") as f:
            await f.write(contents)
        # probabilities, prediction = predict_image(file.filename)
        # return {"filename": file.filename, "prediction": prediction, "probabilities": probabilities.tolist()}
        prediction = predict_image(file.filename)
        return {"filename": file.filename, "prediction": prediction}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500) from e