import logging

import hydra
import torch
import typer
from google.cloud import storage
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from face_classification.data import FaceDataset
from face_classification.model import PretrainedResNet34

app = typer.Typer()


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}.")


@app.command()
def evaluate(model_path: str = None, config_name: str = "default_config") -> None:
    """Evaluate a trained model using PyTorch Lightning Trainer."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Evaluating with model: {model_path}")

    with hydra.initialize(config_path="../../configs", version_base=None, job_name="evaluate_model"):
        cfg = hydra.compose(config_name=config_name)

    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.evaluate

    # Define the test dataset and dataloader
    test_dataset = FaceDataset(mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    model = PretrainedResNet34(cfg)
    if model_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        logger.info("Model weights successfully loaded.")
    else:
        logger.info("Model path not provided. Downloading model from GCP bucket.")
        model_path = "models/model_weights_local.pth"
        bucket_name = "face-classification-models"
        source_blob_name = "models/model_weights.pth"
        download_from_gcs(bucket_name, source_blob_name, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        logger.info("Model weights successfully downloaded and loaded from GCP.")

    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(accelerator=hparams.accelerator, logger=WandbLogger(project="face_classification"))

    # Evaluate the model
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    typer.run(evaluate)
