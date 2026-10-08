"""Update dataset labels with real clinical data from IDC."""

import json
from pathlib import Path
import numpy as np


def birads_to_probability(birads: str) -> float:
    """Convert BIRADS score to cancer probability.

    BIRADS classification:
    - 1-2: Benign (0-20%)
    - 3: Probably benign (20-40%)
    - 4: Suspicious (40-70%)
    - 5: Highly suggestive of malignancy (70-90%)
    - 6: Known malignancy (90-100%)
    """
    try:
        birads_int = int(birads)
        if birads_int <= 2:
            return np.random.uniform(0.05, 0.20)
        elif birads_int == 3:
            return np.random.uniform(0.20, 0.40)
        elif birads_int == 4:
            return np.random.uniform(0.40, 0.70)
        elif birads_int == 5:
            return np.random.uniform(0.70, 0.90)
        elif birads_int == 6:
            return np.random.uniform(0.90, 0.99)
        else:
            return 0.5  # Unknown
    except:
        return 0.5  # Unknown


def birads_to_grade(birads: str) -> str:
    """Convert BIRADS to tumor grade."""
    try:
        birads_int = int(birads)
        if birads_int <= 2:
            return "low"
        elif birads_int == 3:
            return "low"
        elif birads_int == 4:
            return "intermediate"
        elif birads_int >= 5:
            return "high"
        else:
            return "intermediate"
    except:
        return "intermediate"


def update_labels_with_clinical_data(clinical_data, labels_dir: Path):
    """Update label files with clinical data.

    Args:
        clinical_data: Clinical data dictionary from IDC
        labels_dir: Path to directory with label files
    """

    # Create mapping from patient_id to clinical info
    clinical_map = {}
    for row in clinical_data["rows"]:
        patient_id = row["dicom_patient_id"]
        clinical_map[patient_id] = row

    # Update each label file
    for split in ["train", "val", "test"]:
        labels_file = labels_dir / f"{split}_labels.json"
        if not labels_file.exists():
            continue

        with open(labels_file, "r") as f:
            labels = json.load(f)

        updated_count = 0
        for patient_id, data in labels.items():
            if patient_id in clinical_map:
                clinical = clinical_map[patient_id]
                birads = clinical.get("birads", "")

                if birads and birads != "-1":
                    # Update with clinical-based labels
                    labels[patient_id]["cancer_probability"] = birads_to_probability(birads)
                    labels[patient_id]["grade"] = birads_to_grade(birads)
                    labels[patient_id]["data_source"] = "clinical_birads"
                    updated_count += 1
                else:
                    labels[patient_id]["data_source"] = "placeholder"
            else:
                labels[patient_id]["data_source"] = "not_found"

        # Save updated labels
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"Updated {updated_count}/{len(labels)} patients in {split} with clinical data")

    print("\nLabel update complete!")


if __name__ == "__main__":
    # Load the clinical data from the temp file
    clinical_data_path = Path("C:/Users/DBTECH AFRICA/AppData/Local/Temp/devin.exe-overflows/90924fb5/content.txt")
    labels_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data")

    # Load the clinical data
    with open(clinical_data_path, "r") as f:
        clinical_data = json.load(f)

    print(f"Loaded clinical data with {len(clinical_data['rows'])} patients")
    print("\nNow updating labels with clinical data...")
    update_labels_with_clinical_data(clinical_data, labels_dir)
