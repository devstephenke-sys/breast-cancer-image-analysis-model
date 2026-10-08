"""Test preprocessing pipeline on downloaded data."""

import torch
from pathlib import Path
import json

from breast_cancer_ai.dataset import BreastMRIDataset
from breast_cancer_ai.preprocessing import BreastMRI3DPreprocessor


def test_preprocessing():
    """Test preprocessing on a sample of downloaded data."""
    # Paths
    data_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data")
    train_labels = data_dir / "train_labels.json"

    print("Loading dataset...")
    dataset = BreastMRIDataset(
        train_labels,
        target_size=(128, 128, 16),
        augment=False,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        try:
            image, labels = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Labels: {labels}")
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")

    # Test data loader
    print("\nTesting data loader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1].keys()}
        )
    )

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= 2:
            break
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")

    print("\nPreprocessing test complete!")


if __name__ == "__main__":
    test_preprocessing()
