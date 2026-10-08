"""Match downloaded patients with clinical data using DICOM PatientID."""

import json
from pathlib import Path
import pydicom
from collections import defaultdict


def extract_dicom_patient_id(dicom_path: str) -> str:
    """Extract PatientID from a DICOM file."""
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return ds.PatientID
    except Exception as e:
        print(f"Error reading {dicom_path}: {e}")
        return None


def match_patients(labels_dir: Path, idc_data_dir: Path):
    """Match downloaded patients with clinical data by extracting DICOM PatientID.

    Args:
        labels_dir: Path to directory with label files
        idc_data_dir: Path to IDC downloaded data
    """
    # Load current labels
    patient_to_files = defaultdict(list)
    for split in ["train", "val", "test"]:
        labels_file = labels_dir / f"{split}_labels.json"
        if not labels_file.exists():
            continue

        with open(labels_file, "r") as f:
            labels = json.load(f)

        for patient_id, data in labels.items():
            for dicom_file in data.get("dicom_files", []):
                patient_to_files[patient_id].append(dicom_file)

    print(f"Found {len(patient_to_files)} patients in labels")

    # Extract PatientID from first DICOM file for each patient
    patient_id_mapping = {}
    for patient_dir, dicom_files in patient_to_files.items():
        if dicom_files:
            # Try first file
            dicom_patient_id = extract_dicom_patient_id(dicom_files[0])
            if dicom_patient_id:
                patient_id_mapping[patient_dir] = dicom_patient_id
                print(f"{patient_dir} -> DICOM PatientID: {dicom_patient_id}")

    # Save mapping
    mapping_file = labels_dir / "patient_id_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(patient_id_mapping, f, indent=2)

    print(f"\nSaved patient ID mapping to: {mapping_file}")
    print(f"Matched {len(patient_id_mapping)} patients")

    return patient_id_mapping


if __name__ == "__main__":
    labels_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data")
    idc_data_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/idc-data/advanced_mri_breast_lesions")

    patient_id_mapping = match_patients(labels_dir, idc_data_dir)
