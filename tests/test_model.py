import hydra
import torch
from pytorch_lightning import Trainer

from face_classification.model import PretrainedResNet34

with hydra.initialize(config_path="../configs", version_base=None, job_name="train_model"):
    config = hydra.compose(config_name="default_config")


class TestModel:
    """Class that collects all tests concerning the model."""

    def test_model_output(self):
        """Testing model."""
        model = PretrainedResNet34(config)
        assert isinstance(model, torch.nn.Module)

        image = torch.randn(1, 3, 256, 256)
        output = model(image)
        assert output.shape == torch.Size([1, 16]), "Expected output shape to be [1, 16]"

    def test_fine_tuning(self):
        """Testing fine-tuning behavior."""
        model = PretrainedResNet34(config)

        # Check that all parameters except BatchNorm layers are frozen
        for name, param in model.model.named_parameters():
            if not ("bn" in name) and not ("fc" in name):  # Allow BatchNorm layers to remain trainable
                assert not param.requires_grad, f"Expected {name} to be frozen"

        # Check that fully connected layer parameters are trainable
        fc_layer_params = [p.requires_grad for p in model.model.fc.parameters()]
        assert all(fc_layer_params), "Expected fully connected layer parameters to be trainable"

    def test_non_fine_tuning(self):
        """Testing non-fine-tuning behavior."""
        config_no_fine_tuning = config
        config_no_fine_tuning.model.fine_tuning = False
        model = PretrainedResNet34(config_no_fine_tuning)

        all_params = [p.requires_grad for p in model.model.parameters()]
        assert all(all_params), "Expected all parameters to be trainable in non-fine-tuning mode"

    def test_loss_computation(self):
        """Testing loss computation."""
        model = PretrainedResNet34(config)
        model.criterion = torch.nn.CrossEntropyLoss()

        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 256, 256)
        dummy_targets = torch.randint(0, 16, (batch_size,))

        output = model(dummy_images)
        loss = model.criterion(output, dummy_targets)
        assert loss.item() > 0, "Loss should be a positive value"

    def test_gradient_flow(self):
        """Testing gradient flow through the model."""
        model = PretrainedResNet34(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dummy_images = torch.randn(4, 3, 256, 256)
        dummy_targets = torch.randint(0, 16, (4,))
        output = model(dummy_images)

        loss = model.criterion(output, dummy_targets)
        loss.backward()
        optimizer.step()

        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(
            g is not None and g.abs().sum().item() > 0 for g in gradients
        ), "Gradients should flow through the model"


class TestTrainingProcess:
    """Class that collects all tests concerning the model training process."""

    def test_trainer_run_and_parameter_update(self):
        """Test the Trainer runs correctly and updates model parameters."""

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size):
                self.size = size
                self.data = torch.randn(size, 3, 256, 256)
                self.labels = torch.randint(0, 16, (size,))

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]

        # Create datasets and DataLoader
        train_set = DummyDataset(size=8)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=4, num_workers=0)
        val_set = DummyDataset(size=4)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=2, num_workers=0)

        # Initialize the model and move it to the correct device
        model = PretrainedResNet34(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Save a copy of the initial model parameters
        initial_params = {
            name: param.clone().to(device) for name, param in model.named_parameters() if param.requires_grad
        }

        # Create a Trainer and fit the model
        trainer = Trainer(
            max_epochs=1, limit_train_batches=2, limit_val_batches=0, log_every_n_steps=2, accelerator=device.type
        )  # Limit for quick testing
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

        # Check if any of the trainable parameters have changed
        updated_params = {name: param.to(device) for name, param in model.named_parameters() if param.requires_grad}
        has_changed = any(not torch.equal(initial_params[name], updated_params[name]) for name in initial_params)

        assert has_changed, "Model parameters should have changed after training"
