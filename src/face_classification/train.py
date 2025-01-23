import logging
import os

import hydra
import torch
import typer
import wandb
from google.cloud import storage
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from face_classification.data import FaceDataset
from face_classification.metric_tracker import MetricTracker
from face_classification.model import PretrainedResNet34

app = typer.Typer()
import wandb

wandb.login()


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def save_model_to_onnx(model, checkpoint_path, output_path):
    """Loads the best PyTorch checkpoint and converts it to ONNX."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])  # Load the checkpoint weights
    model.eval()  # Ensure the model is in evaluation mode

    dummy_input = torch.randn(1, 3, 256, 256, device="cpu")  # Adjust input dimensions as needed
    model.to_onnx(output_path, dummy_input, export_params=True)

    print(f"Model successfully exported to {output_path}")


def save_pytorch_model_weights(model, checkpoint_path, output_path):
    """Saves the PyTorch model weights."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])  # Load the checkpoint weights
    torch.save(model.state_dict(), output_path)
    print(f"PyTorch model weights saved to {output_path}")


@app.command()
def train(config_name: str = "default_config") -> None:
    """Train a model on the Face Dataset."""

    # Initialize Hydra configuration
    with hydra.initialize(config_path="../../configs", version_base=None, job_name="train_model"):
        cfg = hydra.compose(config_name=config_name)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    hparams = cfg.train

    if hparams.use_tensorflow_profiler:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(
                f"reports/profiler_logs/tensorboard/PretrainedResNet34_train_{hparams.epochs}_epoch"
            ),
        ) as prof:
            run_training(cfg, hparams)
        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    if hparams.use_chromium_profiler:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            run_training(cfg, hparams)
        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        logger.info(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
        logdir = f"reports/profiler_logs/chromium_trace/PretrainedResNet34_train_{hparams.epochs}_epoch"
        os.makedirs(logdir, exist_ok=True)
        filename = f"{logdir}/train_trace.json"
        prof.export_chrome_trace(filename)
    if not hparams.use_chromium_profiler and not hparams.use_tensorflow_profiler:
        run_training(cfg, hparams)


def run_training(cfg, hparams) -> None:
    run = wandb.init(
        entity="face_classification", project="face_classification", config=OmegaConf.to_container(hparams)
    )  # type: ignore
    wandb.log(OmegaConf.to_container(hparams))  # type: ignore

    train_set = FaceDataset(mode="train")
    val_set = FaceDataset(mode="val")
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=False
    )

    model = PretrainedResNet34(cfg)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        dirpath="models/checkpoints/",
        filename="model-{epoch:02d}-{val_acc:.2f}",
        save_top_k=hparams.save_top_k,
        mode=hparams.mode,
    )
    # Get the first batch from the validation dataloader
    first_val_batch = next(iter(val_dataloader))
    # Reset the validation dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=False
    )
    metric_tracker = MetricTracker(first_val_batch, num_samples=cfg.evaluate.batch_size)

    trainer = Trainer(
        max_epochs=hparams.epochs,
        callbacks=[checkpoint_callback, metric_tracker],
        accelerator="auto",
        logger=WandbLogger(project="face_classification"),
    )
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = checkpoint_callback.best_model_path

    # Log the model to W&B registry
    if best_model_path:  # Ensure a model checkpoint exists
        # Save the ONNX model
        onnx_model_path = "models/model_final.onnx"
        save_model_to_onnx(model, best_model_path, onnx_model_path)

        # Save the PyTorch model weights
        pytorch_weights_path = "models/model_weights.pth"
        save_pytorch_model_weights(model, best_model_path, pytorch_weights_path)

        # Upload models to GCP
        bucket_name = "face-classification-models"
        upload_to_gcs(bucket_name, onnx_model_path, "models/model_final.onnx")
        upload_to_gcs(bucket_name, pytorch_weights_path, "models/model_weights.pth")

        artifact = wandb.Artifact(
            name="face_classification_model",
            type="model",
            description="A model trained to classify face images",
        )
        artifact.add_file(best_model_path)  # Add the saved model file
        run.log_artifact(artifact)  # Log artifact to W&B

        # Link the artifact to the model registry
        artifact.wait()  # Ensure artifact is fully uploaded before linking
        artifact.aliases.append("latest")  # Add alias for the artifact version
        artifact.aliases.append(f"v{artifact.version}")

        # Link the artifact to the registry
        artifact.link(
            target_path=cfg.urls.wandb_registry
        )


if __name__ == "__main__":
    typer.run(train)
