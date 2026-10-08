"""Download ACRIN-6698 breast MRI data from IDC.

This collection has 385 patients with comprehensive clinical data including:
- Lesion type (malignant/benign)
- Hormone receptor status
- Tumor grade
- Treatment response (pCR)
"""

from idc_index import index as idc_index
from pathlib import Path
import sys


def download_acrin6698(output_dir: str, max_patients: int = 100):
    """Download ACRIN-6698 breast MRI data.

    Args:
        output_dir: Directory to save downloaded data
        max_patients: Maximum number of patients to download (default: 100)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ACRIN-6698 data to: {output_path}")
    print(f"Target: {max_patients} patients")

    # Build cohort for ACRIN-6698 with MR modality
    # Filter for MR modality and breast location
    cohort = idc_index.build_cohort(
        collection_id="acrin_6698",
        Modality=["MR"],
        BodyPartExamined=["BREAST"]
    )

    print(f"\nCohort statistics:")
    print(f"  Patients: {cohort.counts.patients}")
    print(f"  Studies: {cohort.counts.studies}")
    print(f"  Series: {cohort.counts.series}")
    print(f"  Size: {cohort.counts.size_TB:.2f} TB")

    # Get series UIDs for download
    # Limit to max_patients by sampling from the cohort
    series_uids = cohort.series_instance_uids[:max_patients * 20]  # Approx 20 series per patient

    print(f"\nStarting download of {len(series_uids)} series...")
    print("This may take a while...")

    # Download using IDC index
    try:
        idc_index.download(
            series_instance_uids=series_uids,
            download_dir=str(output_path)
        )
        print(f"\nDownload complete!")
        print(f"Data saved to: {output_path}")
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    output_dir = "C:/Users/DBTECH AFRICA/Desktop/Cancer Research/idc-data/acrin_6698"
    max_patients = 100  # Start with 100 patients

    download_acrin6698(output_dir, max_patients)
