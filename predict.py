"""Run inference on new DICOM scans."""

import torch
import pydicom
import numpy as np
from pathlib import Path
from PIL import Image


class SimpleCNN(torch.nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 128)
        self.fc_cancer = torch.nn.Linear(128, 1)
        self.fc_grade = torch.nn.Linear(128, 3)
        self.fc_response = torch.nn.Linear(128, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return {
            "cancer": torch.sigmoid(self.fc_cancer(x)),
            "grade": self.fc_grade(x),
            "response": self.fc_response(x),
        }


def load_dicom_image(dicom_path: str, target_size=(256, 256)):
    """Load and preprocess DICOM image.

    Args:
        dicom_path: Path to DICOM file
        target_size: Target size for image

    Returns:
        Preprocessed image tensor
    """
    # Handle long paths on Windows
    import os
    if len(dicom_path) > 260 and not dicom_path.startswith("\\\\?\\"):
        dicom_path = "\\\\?\\" + os.path.abspath(dicom_path)

    ds = pydicom.dcmread(dicom_path)

    if hasattr(ds, 'pixel_array'):
        image = ds.pixel_array
    else:
        image = np.zeros(target_size, dtype=np.float32)

    # Normalize
    if len(image.shape) == 2:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Resize
        if image.shape != target_size:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
            image_pil = image_pil.resize(target_size)
            image = np.array(image_pil).astype(np.float32) / 255.0

    # Convert to tensor
    image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
    return image.unsqueeze(0)  # Add batch dimension


def predict(model, dicom_path: str, device: str = "cpu"):
    """Run prediction on a DICOM file.

    Args:
        model: Trained model
        dicom_path: Path to DICOM file
        device: Device to run inference on

    Returns:
        Dictionary with predictions
    """
    model.eval()
    image = load_dicom_image(dicom_path).to(device)

    with torch.no_grad():
        outputs = model(image)

    # Get predictions
    cancer_prob = outputs["cancer"].item()
    grade_idx = outputs["grade"].argmax().item()
    response_idx = outputs["response"].argmax().item()

    # Convert to labels
    grade_labels = ["low", "intermediate", "high"]
    response_labels = ["complete", "partial", "no_response"]

    return {
        "cancer_probability": cancer_prob,
        "grade": grade_labels[grade_idx],
        "treatment_response": response_labels[response_idx],
    }


def main():
    """Main inference function."""
    # Load model
    model_path = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully")

    # Get a test DICOM file
    test_labels_file = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/test_labels.json")

    import json
    with open(test_labels_file, "r") as f:
        test_labels = json.load(f)

    # Get first patient's first DICOM file
    patient_id = list(test_labels.keys())[0]
    dicom_path = test_labels[patient_id]["dicom_files"][0]

    # Clean path (remove leading backslash if present)
    if dicom_path.startswith("\\"):
        dicom_path = dicom_path[1:]

    print(f"\nRunning prediction on: {patient_id}")
    print(f"DICOM file: {dicom_path}")

    # Run prediction
    result = predict(model, dicom_path, device)

    print("\n=== Prediction Results ===")
    print(f"Cancer Probability: {result['cancer_probability']:.4f}")
    print(f"Grade: {result['grade']}")
    print(f"Treatment Response: {result['treatment_response']}")


if __name__ == "__main__":
    main()
