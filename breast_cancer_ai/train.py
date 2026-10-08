"""Training script for breast cancer classification model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import json
from pathlib import Path

from .model import BreastCancerClassifier
from .preprocessing import get_preprocessor


class BreastCancerTrainer:
    """Trainer for breast cancer classification model."""

    def __init__(
        self,
        model: BreastCancerClassifier,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss functions for each task
        self.binary_criterion = nn.CrossEntropyLoss()
        self.grade_criterion = nn.CrossEntropyLoss()
        self.receptor_criterion = nn.BCEWithLogitsLoss()
        self.response_criterion = nn.CrossEntropyLoss()

        # Task weights (can be adjusted)
        self.task_weights = {
            "binary": 1.0,
            "grade": 0.5,
            "receptor": 0.3,
            "response": 0.3,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            outputs: Model outputs
            labels: Ground truth labels

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Binary classification loss
        if "binary" in labels:
            losses["binary"] = self.binary_criterion(outputs["binary"], labels["binary"])

        # Grade classification loss
        if "grade" in labels:
            losses["grade"] = self.grade_criterion(outputs["grade"], labels["grade"])

        # Receptor status loss
        if "receptor" in labels:
            losses["receptor"] = self.receptor_criterion(outputs["receptor"], labels["receptor"])

        # Treatment response loss
        if "response" in labels:
            losses["response"] = self.response_criterion(outputs["response"], labels["response"])

        # Total weighted loss
        total_loss = sum(
            self.task_weights.get(task, 0.0) * loss
            for task, loss in losses.items()
        )
        losses["total"] = total_loss

        return losses

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses = {key: 0.0 for key in self.task_weights.keys()}
        epoch_losses["total"] = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute loss
            losses = self.compute_loss(outputs, labels)

            # Backward pass
            losses["total"].backward()
            self.optimizer.step()

            # Accumulate losses
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()

        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct_binary = 0
        correct_grade = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                outputs = self.model(images)
                losses = self.compute_loss(outputs, labels)

                total_loss += losses["total"].item()

                # Compute accuracy
                if "binary" in labels:
                    binary_pred = torch.argmax(outputs["binary"], dim=1)
                    correct_binary += (binary_pred == labels["binary"]).sum().item()

                if "grade" in labels:
                    grade_pred = torch.argmax(outputs["grade"], dim=1)
                    correct_grade += (grade_pred == labels["grade"]).sum().item()

                total_samples += images.size(0)

        metrics = {
            "val_loss": total_loss / len(val_loader),
            "binary_accuracy": correct_binary / total_samples if total_samples > 0 else 0.0,
            "grade_accuracy": correct_grade / total_samples if total_samples > 0 else 0.0,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        early_stopping_patience: int = 10,
    ):
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Epoch {epoch}: {val_metrics}")

                # Early stopping
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0

                    # Save best model
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    }, save_dir / "best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                print(f"Epoch {epoch}: {train_losses}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, save_dir / f"checkpoint_epoch_{epoch}.pth")

        print("Training complete!")
