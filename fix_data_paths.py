"""Fix DICOM file paths in labels JSON to match actual data location."""

import json
from pathlib import Path
import glob

def fix_labels_file(labels_path, data_dir):
    """Fix paths in labels JSON file - use actual DICOM files found on disk."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    fixed_count = 0
    for patient_id, data in labels.items():
        # Find the actual patient directory
        patient_dir = data_dir / patient_id
        if not patient_dir.exists():
            print(f"Patient directory not found: {patient_id}")
            continue
        
        # Find all DICOM files for this patient
        actual_files = list(patient_dir.rglob('*.dcm'))
        
        if actual_files:
            # Update with actual file paths (keep Windows backslashes)
            data['dicom_files'] = [str(f) for f in actual_files]
            fixed_count += 1
            print(f"Fixed {len(actual_files)} files for {patient_id}")
        else:
            print(f"No DICOM files found for {patient_id}")
    
    # Save fixed labels
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\nFixed {fixed_count} patients in {labels_path}")

if __name__ == "__main__":
    data_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/idc-data/advanced_mri_breast_lesions")
    
    for split in ['train', 'val', 'test']:
        labels_path = Path(f"C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/{split}_labels.json")
        if labels_path.exists():
            print(f"\nFixing {split} labels...")
            fix_labels_file(labels_path, data_dir)
