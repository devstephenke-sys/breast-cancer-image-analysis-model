"""Remove patients with missing DICOM files from labels."""

import json
from pathlib import Path

def cleanup_labels(labels_path):
    """Remove patients with missing files."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    valid_patients = {}
    for patient_id, data in labels.items():
        dicom_files = data.get('dicom_files', [])
        if not dicom_files:
            continue
        
        # Check if at least one file exists
        has_valid_file = False
        for file_path in dicom_files:
            if Path(file_path).exists():
                has_valid_file = True
                break
        
        if has_valid_file:
            valid_patients[patient_id] = data
        else:
            print(f"Removing {patient_id} - no valid files")
    
    # Save cleaned labels
    with open(labels_path, 'w') as f:
        json.dump(valid_patients, f, indent=2)
    
    print(f"Saved {len(valid_patients)} valid patients (removed {len(labels) - len(valid_patients)})")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        labels_path = Path(f"C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/{split}_labels.json")
        if labels_path.exists():
            print(f"\nCleaning {split} labels...")
            cleanup_labels(labels_path)
