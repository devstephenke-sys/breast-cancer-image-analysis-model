"""Run training on the prepared dataset."""

import torch
from pathlib import Path
from tqdm import tqdm

from breast_cancer_ai.dataset import create_data_loaders
from breast_cancer_ai.model import BreastCancerClassifier
from breast_cancer_ai.train import BreastCancerTrainer


def train_model():
    """Train the breast cancer classification model with extended training."""
    # Paths
    data_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data")
    checkpoint_dir = data_dir / "checkpoints_extended"

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=1,  # Very small batch size for 3D MRI
        num_workers=0,  # Set to 0 for Windows compatibility
        modality="MR",
        target_size=(128, 128, 64),  # Higher resolution for better training
        augment_train=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use a deeper 3D CNN for better learning
    import torch.nn as nn
    
    class Deep3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv3d(1, 32, 3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 32, 3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Dropout3d(0.2),
                
                # Block 2
                nn.Conv3d(32, 64, 3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, 64, 3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Dropout3d(0.2),
                
                # Block 3
                nn.Conv3d(64, 128, 3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.Conv3d(128, 128, 3, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Dropout3d(0.3),
                
                # Block 4
                nn.Conv3d(128, 256, 3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            self.fc = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            
            # Task-specific heads
            self.head_binary = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2),
            )
            self.head_grade = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3),
            )
            self.head_receptor = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3),
            )
            self.head_response = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 3),
            )
        
        def forward(self, x):
            features = self.features(x)
            features = features.view(features.size(0), -1)
            features = self.fc(features)
            
            return {
                "binary": self.head_binary(features),
                "grade": self.head_grade(features),
                "receptor": self.head_receptor(features),
                "response": self.head_response(features),
            }
    
    model = Deep3DCNN().to(device)

    # Create trainer
    trainer = BreastCancerTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
    )

    # Train with extended epochs for more thorough learning
    print("\nStarting extended training (50 epochs)...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,  # Increased from 10 to 50 for more thorough training
        save_dir=str(checkpoint_dir),
        early_stopping_patience=15,  # Increased patience to allow more training
    )

    print("\nTraining complete!")
    print(f"Model checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    train_model()
