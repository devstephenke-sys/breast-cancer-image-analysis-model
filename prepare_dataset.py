"""Prepare dataset from downloaded IDC DICOM files."""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pydicom
import numpy as np
from tqdm import tqdm


def scan_dicom_directory(data_dir: Path) -> Dict[str, List[Path]]:
    """Scan directory for DICOM files organized by patient.

    Args:
        data_dir: Root directory containing patient folders

    Returns:
        Dictionary mapping patient_id to list of DICOM file paths
    """
    patient_dict = {}

    print(f"Scanning directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}")

    if not data_dir.exists():
        print(f"ERROR: Directory does not exist: {data_dir}")
        return patient_dict

    # List all items in directory
    items = list(data_dir.iterdir())
    print(f"Found {len(items)} items in directory")

    for patient_dir in items:
        print(f"Checking: {patient_dir.name}, is_dir: {patient_dir.is_dir()}")
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        dicom_files = []

        # Use scandir for better Windows compatibility with long path support
        dicom_files = []
        try:
            # Convert to absolute path and add Windows long path prefix
            patient_dir_abs = os.path.abspath(str(patient_dir))
            if len(patient_dir_abs) > 0:
                walk_path = r"\\?\\" + patient_dir_abs
            else:
                walk_path = patient_dir_abs

            for root, dirs, files in os.walk(walk_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                            # Store without prefix for compatibility
                            clean_path = file_path
                            if clean_path.startswith("\\\\?\\"):
                                clean_path = clean_path[4:]
                            # Also remove any leading backslash
                            if clean_path.startswith("\\"):
                                clean_path = clean_path.lstrip("\\")
                            dicom_files.append(Path(clean_path))
                    except Exception as e:
                        if len(dicom_files) == 0:  # Only print first error
                            print(f"    First error: {e}")
                        pass
        except Exception as e:
            print(f"    Walk error: {e}")

        print(f"  Found {len(dicom_files)} valid files")

        if dicom_files:
            patient_dict[patient_id] = dicom_files
            print(f"Found {len(dicom_files)} files for patient {patient_id}")

    return patient_dict


def create_dataset_structure(
    patient_dict: Dict[str, List[Path]],
    output_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Tuple[Dict, Dict, Dict]:
    """Create train/val/test splits and organize dataset.

    Args:
        patient_dict: Dictionary of patient_id to DICOM files
        output_dir: Output directory for organized dataset
        train_split: Fraction for training
        val_split: Fraction for validation

    Returns:
        Tuple of (train, val, test) patient dictionaries
    """
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Split patients
    patient_ids = list(patient_dict.keys())
    np.random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    train_dict = {pid: patient_dict[pid] for pid in train_ids}
    val_dict = {pid: patient_dict[pid] for pid in val_ids}
    test_dict = {pid: patient_dict[pid] for pid in test_ids}

    return train_dict, val_dict, test_dict


def create_labels_json(
    train_dict: Dict,
    val_dict: Dict,
    test_dict: Dict,
    output_dir: Path,
):
    """Create labels JSON file for training.

    Since clinical labels are not in DICOM files, we create placeholder labels.
    In production, these would come from clinical data tables via IDC.
    """
    def create_labels_for_split(split_dict: Dict, split_name: str):
        labels = {}
        for patient_id, dicom_files in split_dict.items():
            # Placeholder labels - in production, fetch from clinical data
            # These would come from IDC clinical tables for advanced_mri_breast_lesions
            labels[patient_id] = {
                "cancer_probability": np.random.random(),  # Placeholder
                "grade": np.random.choice(["low", "intermediate", "high"]),
                "receptor_status": {
                    "ER": np.random.choice(["positive", "negative"]),
                    "PR": np.random.choice(["positive", "negative"]),
                    "HER2": np.random.choice(["positive", "negative"]),
                },
                "ki67": np.random.randint(0, 100),
                "treatment_response": np.random.choice(["complete", "partial", "no_response"]),
                "dicom_files": [str(f).replace("\\\\", "\\") for f in dicom_files],
            }

        # Save labels
        labels_file = output_dir / f"{split_name}_labels.json"
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"Created {split_name} labels: {len(labels)} patients")

    create_labels_for_split(train_dict, "train")
    create_labels_for_split(val_dict, "val")
    create_labels_for_split(test_dict, "test")


def verify_dicom_files(patient_dict: Dict[str, List[Path]]) -> int:
    """Verify DICOM files can be read.

    Args:
        patient_dict: Dictionary of patient_id to DICOM files

    Returns:
        Number of valid DICOM files
    """
    valid_count = 0
    total_count = 0

    for patient_id, dicom_files in patient_dict.items():
        for dcm_file in dicom_files:
            total_count += 1
            try:
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'pixel_array'):
                    valid_count += 1
            except Exception as e:
                print(f"Error reading {dcm_file}: {e}")

    print(f"Verified {valid_count}/{total_count} DICOM files")
    return valid_count


def main():
    """Main dataset preparation function."""
    # Paths
    data_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/idc-data/advanced_mri_breast_lesions")
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning DICOM directory...")
    patient_dict = scan_dicom_directory(data_dir)
    print(f"Found {len(patient_dict)} patients with DICOM files")

    # Verify DICOM files
    verify_dicom_files(patient_dict)

    # Create train/val/test splits
    print("\nCreating train/val/test splits...")
    train_dict, val_dict, test_dict = create_dataset_structure(patient_dict, output_dir)

    print(f"Train: {len(train_dict)} patients")
    print(f"Val: {len(val_dict)} patients")
    print(f"Test: {len(test_dict)} patients")

    # Create labels
    print("\nCreating labels...")
    create_labels_json(train_dict, val_dict, test_dict, output_dir)

    # Save dataset info
    dataset_info = {
        "n_train": len(train_dict),
        "n_val": len(val_dict),
        "n_test": len(test_dict),
        "total_patients": len(patient_dict),
        "data_source": "NCI IDC - Advanced-MRI-Breast-Lesions",
        "notes": "Labels are placeholders - need to fetch clinical data from IDC for real labels",
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\nDataset preparation complete!")
    print(f"Data organized in: {output_dir}")


if __name__ == "__main__":
    main()
