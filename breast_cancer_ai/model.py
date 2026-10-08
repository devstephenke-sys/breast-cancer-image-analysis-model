"""Breast Cancer Classification Model Architecture."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from monai.networks.nets import DenseNet121, ResNet18


class BreastCancerClassifier(nn.Module):
    """Multi-task breast cancer classification model.

    Tasks:
    1. Binary classification: Malignant vs Benign
    2. Grade classification: Low, Intermediate, High
    3. Receptor status: ER, PR, HER2
    4. Treatment response: Complete, Partial, No response
    """

    def __init__(
        self,
        in_channels: int = 1,
        spatial_dims: int = 3,
        num_classes_grade: int = 3,
        num_classes_response: int = 3,
        pretrained: bool = True,
    ):
        """Initialize the model.

        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            spatial_dims: Spatial dimensions (2 for 2D images, 3 for 3D volumes)
            num_classes_grade: Number of grade classes
            num_classes_response: Number of treatment response classes
            pretrained: Use pretrained weights
        """
        super().__init__()

        self.spatial_dims = spatial_dims

        # Backbone: 3D DenseNet121 for MRI, 2D ResNet18 for mammography
        if spatial_dims == 3:
            # 3D CNN for MRI volumes
            self.backbone = DenseNet121(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=512,
                pretrained=pretrained,
            )
            feature_dim = 512
        else:
            # 2D CNN for mammography
            self.backbone = ResNet18(
                spatial_dims=2,
                in_channels=in_channels,
                num_classes=512,
                pretrained=pretrained,
            )
            feature_dim = 512

        # Task-specific heads
        # 1. Binary classification: Malignant vs Benign
        self.head_binary = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

        # 2. Grade classification: Low, Intermediate, High
        self.head_grade = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_grade),
        )

        # 3. Receptor status: ER, PR, HER2 (each binary)
        self.head_receptor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),  # ER, PR, HER2
        )

        # 4. Treatment response: Complete, Partial, No response
        self.head_response = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_response),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, *spatial_dims)

        Returns:
            Dictionary of predictions for each task
        """
        # Extract features
        features = self.backbone(x)

        # Task-specific predictions
        binary_logits = self.head_binary(features)
        grade_logits = self.head_grade(features)
        receptor_logits = self.head_receptor(features)
        response_logits = self.head_response(features)

        return {
            "binary": binary_logits,
            "grade": grade_logits,
            "receptor": receptor_logits,
            "response": response_logits,
        }


class BreastCancerPredictor:
    """Predictor class for inference."""

    def __init__(self, model: BreastCancerClassifier, device: str = "cuda"):
        """Initialize predictor.

        Args:
            model: Trained model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.grade_labels = ["Low", "Intermediate", "High"]
        self.response_labels = ["Complete", "Partial", "No Response"]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, any]:
        """Make predictions.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and probabilities
        """
        x = x.to(self.device)
        outputs = self.model(x)

        # Convert logits to probabilities
        binary_probs = torch.softmax(outputs["binary"], dim=1)
        grade_probs = torch.softmax(outputs["grade"], dim=1)
        receptor_probs = torch.sigmoid(outputs["receptor"])
        response_probs = torch.softmax(outputs["response"], dim=1)

        # Get predictions
        binary_pred = torch.argmax(binary_probs, dim=1)
        grade_pred = torch.argmax(grade_probs, dim=1)
        response_pred = torch.argmax(response_probs, dim=1)

        return {
            "cancer_probability": binary_probs[:, 1].cpu().item(),
            "is_malignant": binary_pred.cpu().item() == 1,
            "grade": self.grade_labels[grade_pred.cpu().item()],
            "grade_probabilities": {
                label: prob.cpu().item()
                for label, prob in zip(self.grade_labels, grade_probs[0])
            },
            "receptor_status": {
                "ER_positive": receptor_probs[0, 0].cpu().item() > 0.5,
                "PR_positive": receptor_probs[0, 1].cpu().item() > 0.5,
                "HER2_positive": receptor_probs[0, 2].cpu().item() > 0.5,
            },
            "treatment_response": self.response_labels[response_pred.cpu().item()],
            "response_probabilities": {
                label: prob.cpu().item()
                for label, prob in zip(self.response_labels, response_probs[0])
            },
        }

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cuda") -> "BreastCancerPredictor":
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on

        Returns:
            BreastCancerPredictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Determine spatial_dims from checkpoint
        spatial_dims = checkpoint.get("spatial_dims", 3)

        model = BreastCancerClassifier(
            spatial_dims=spatial_dims,
            num_classes_grade=checkpoint.get("num_classes_grade", 3),
            num_classes_response=checkpoint.get("num_classes_response", 3),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device)
