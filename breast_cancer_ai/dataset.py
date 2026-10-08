"""Dataset class for loading breast cancer DICOM data."""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pydicom
import numpy as np
from torch.utils.data import Dataset

from .preprocessing import BreastMRI3DPreprocessor, Mammography2DPreprocessor


class BreastCancerDataset(Dataset):
    """Dataset for breast cancer DICOM images."""

    def __init__(
        self,
        labels_file: Path,
        modality: str = "MR",
        target_size: Optional[Tuple] = None,
        augment: bool = False,
    ):
        """Initialize dataset.

        Args:
            labels_file: Path to JSON file with labels and DICOM file paths
            modality: Imaging modality (MR, MG, etc.)
            target_size: Target size for preprocessing
            augment: Whether to apply data augmentation
        """
        self.modality = modality
        self.augment = augment

        # Load labels
        with open(labels_file, "r") as f:
            self.labels_data = json.load(f)

        # Get preprocessor
        if modality == "MR":
            if target_size is None:
                target_size = (256, 256, 32)
            self.preprocessor = BreastMRI3DPreprocessor(
                target_size=target_size,
                augment=augment,
            )
        elif modality == "MG":
            if target_size is None:
                target_size = (512, 512)
            self.preprocessor = Mammography2DPreprocessor(
                target_size=target_size,
                augment=augment,
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Build index
        self.samples = []
        for patient_id, data in self.labels_data.items():
            dicom_files = data.get("dicom_files", [])
            if dicom_files:
                self.samples.append({
                    "patient_id": patient_id,
                    "dicom_files": dicom_files,
                    "labels": data,
                })

        print(f"Loaded {len(self.samples)} samples from {labels_file}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, labels_dict)
        """
        sample = self.samples[idx]
        dicom_files = sample["dicom_files"]
        labels = sample["labels"]

        # Load DICOM files
        if self.modality == "MR":
            # Load as 3D volume
            image = self.preprocessor.load_mri_series(dicom_files)
        else:
            # Load single 2D image
            image = self.preprocessor.load_dicom(dicom_files[0])

        # Preprocess
        image_tensor = self.preprocessor.transform(image)

        # Convert labels to tensors
        label_tensors = {}

        # Binary classification: malignant vs benign
        cancer_prob = labels.get("cancer_probability", 0.5)
        is_malignant = 1 if cancer_prob > 0.5 else 0
        label_tensors["binary"] = torch.tensor(is_malignant, dtype=torch.long)

        # Grade classification
        grade = labels.get("grade", "intermediate")
        grade_map = {"low": 0, "intermediate": 1, "high": 2}
        label_tensors["grade"] = torch.tensor(grade_map.get(grade, 1), dtype=torch.long)

        # Receptor status
        receptor_status = labels.get("receptor_status", {})
        receptor_positive = [
            1 if receptor_status.get("ER", "negative") == "positive" else 0,
            1 if receptor_status.get("PR", "negative") == "positive" else 0,
            1 if receptor_status.get("HER2", "negative") == "positive" else 0,
        ]
        label_tensors["receptor"] = torch.tensor(receptor_positive, dtype=torch.float32)

        # Treatment response
        response = labels.get("treatment_response", "no_response")
        response_map = {"complete": 0, "partial": 1, "no_response": 2}
        label_tensors["response"] = torch.tensor(response_map.get(response, 2), dtype=torch.long)

        return image_tensor, label_tensors


class BreastMRIDataset(Dataset):
    """Specialized dataset for 3D breast MRI."""

    def __init__(
        self,
        labels_file: Path,
        target_size: Tuple[int, int, int] = (256, 256, 32),
        augment: bool = False,
    ):
        """Initialize MRI dataset.

        Args:
            labels_file: Path to JSON file with labels
            target_size: Target 3D size (H, W, D)
            augment: Whether to apply augmentation
        """
        self.target_size = target_size
        self.augment = augment

        # Load labels
        with open(labels_file, "r") as f:
            self.labels_data = json.load(f)

        # Preprocessor
        self.preprocessor = BreastMRI3DPreprocessor(
            target_size=target_size,
            augment=augment,
        )

        # Build index
        self.samples = []
        for patient_id, data in self.labels_data.items():
            dicom_files = data.get("dicom_files", [])
            if dicom_files:
                self.samples.append({
                    "patient_id": patient_id,
                    "dicom_files": dicom_files,
                    "labels": data,
                })

        print(f"Loaded {len(self.samples)} MRI samples from {labels_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        dicom_files = sample["dicom_files"]
        labels = sample["labels"]

        # Load 3D MRI volume
        try:
            # Fix path issues (remove leading backslash or double backslash)
            fixed_files = []
            for f in dicom_files:
                if isinstance(f, str):
                    # Remove leading backslash or double backslash
                    if f.startswith("\\"):
                        f = f.lstrip("\\")
                    fixed_files.append(str(f))
                else:
                    fixed_files.append(str(f))

            volume = self.preprocessor.load_mri_series(fixed_files)
            image_tensor = self.preprocessor.transform(volume)
        except Exception as e:
            print(f"Error loading {sample['patient_id']}: {e}")
            # Return a blank tensor if loading fails
            image_tensor = torch.zeros((1, *self.target_size))

        # Convert labels to tensors
        label_tensors = {}

        cancer_prob = labels.get("cancer_probability", 0.5)
        is_malignant = 1 if cancer_prob > 0.5 else 0
        label_tensors["binary"] = torch.tensor(is_malignant, dtype=torch.long)

        grade = labels.get("grade", "intermediate")
        grade_map = {"low": 0, "intermediate": 1, "high": 2}
        label_tensors["grade"] = torch.tensor(grade_map.get(grade, 1), dtype=torch.long)

        receptor_status = labels.get("receptor_status", {})
        receptor_positive = [
            1 if receptor_status.get("ER", "negative") == "positive" else 0,
            1 if receptor_status.get("PR", "negative") == "positive" else 0,
            1 if receptor_status.get("HER2", "negative") == "positive" else 0,
        ]
        label_tensors["receptor"] = torch.tensor(receptor_positive, dtype=torch.float32)

        response = labels.get("treatment_response", "no_response")
        response_map = {"complete": 0, "partial": 1, "no_response": 2}
        label_tensors["response"] = torch.tensor(response_map.get(response, 2), dtype=torch.long)

        return image_tensor, label_tensors


def create_data_loaders(
    data_dir: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    modality: str = "MR",
    target_size: Optional[Tuple] = None,
    augment_train: bool = True,
) -> Tuple:
    """Create train, val, and test data loaders.

    Args:
        data_dir: Directory containing train_labels.json, val_labels.json, test_labels.json
        batch_size: Batch size
        num_workers: Number of worker processes
        modality: Imaging modality
        target_size: Target size for preprocessing
        augment_train: Whether to augment training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_labels = data_dir / "train_labels.json"
    val_labels = data_dir / "val_labels.json"
    test_labels = data_dir / "test_labels.json"

    if not train_labels.exists():
        raise FileNotFoundError(f"Train labels not found: {train_labels}")

    # Choose dataset class based on modality
    if modality == "MR":
        DatasetClass = BreastMRIDataset
    else:
        DatasetClass = BreastCancerDataset

    # Create datasets
    train_dataset = DatasetClass(
        train_labels,
        target_size=target_size,
        augment=augment_train,
    )

    val_dataset = DatasetClass(
        val_labels if val_labels.exists() else train_labels,
        target_size=target_size,
        augment=False,
    )

    test_dataset = DatasetClass(
        test_labels if test_labels.exists() else train_labels,
        target_size=target_size,
        augment=False,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function to handle variable-sized tensors."""
    images = []
    labels = {key: [] for key in batch[0][1].keys()}

    for image, label_dict in batch:
        images.append(image)
        for key, value in label_dict.items():
            labels[key].append(value)

    # Stack images
    try:
        images_batch = torch.stack(images)
    except RuntimeError:
        # If stacking fails, pad to max size
        max_size = max(img.shape for img in images)
        padded_images = []
        for img in images:
            if img.shape != max_size:
                padding = [0] * (2 * len(max_size))
                for i, (current, target) in enumerate(zip(img.shape, max_size)):
                    if current < target:
                        padding[2 * i + 1] = target - current
                img = torch.nn.functional.pad(img, padding)
            padded_images.append(img)
        images_batch = torch.stack(padded_images)

    # Stack labels
    labels_batch = {key: torch.stack(vals) for key, vals in labels.items()}

    return images_batch, labels_batch
