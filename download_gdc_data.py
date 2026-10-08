"""Download open-access genomic data from GDC for breast cancer training."""

import json
import requests
from pathlib import Path
from tqdm import tqdm


def download_gdc_genomic_data(output_dir: Path, target_size_gb: float = 8.0):
    """Download open-access genomic data from GDC.

    Args:
        output_dir: Directory to save downloaded data
        target_size_gb: Target download size in GB
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    GDC_API_BASE = "https://api.gdc.cancer.gov"

    print("Searching for open-access breast cancer genomic data...")

    # Search for gene expression files (open access, breast cancer)
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.primary_site",
                    "value": ["Breast"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.access",
                    "value": ["open"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["TSV"]
                }
            }
        ]
    }

    # Search for files
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "file_id",
            "file_name",
            "data_type",
            "data_format",
            "cases.case_id",
            "cases.submitter_id",
            "file_size"
        ]),
        "format": "JSON",
        "size": 100,
        "pretty": "true"
    }

    response = requests.post(
        f"{GDC_API_BASE}/files",
        json=filters,
        params=params
    )
    response.raise_for_status()
    data = response.json()

    files = data["data"]["hits"]
    print(f"Found {len(files)} open-access gene expression files")

    # Calculate total size
    total_size = sum(f.get("file_size", 0) for f in files)
    total_size_gb = total_size / (1024**3)
    print(f"Total size available: {total_size_gb:.2f} GB")

    # Select files to download to reach target size
    selected_files = []
    current_size = 0
    target_bytes = target_size_gb * (1024**3)

    for file_info in files:
        file_size = file_info.get("file_size", 0)
        if current_size + file_size <= target_bytes:
            selected_files.append(file_info)
            current_size += file_size
        # If we've exceeded target, stop
        if current_size >= target_bytes:
            break

    print(f"Selected {len(selected_files)} files ({current_size / (1024**3):.2f} GB)")

    # Download files directly
    print("\nStarting downloads...")
    downloaded_count = 0
    failed_count = 0

    for file_info in tqdm(selected_files, desc="Downloading files"):
        file_id = file_info["file_id"]
        file_name = file_info["file_name"]
        file_size = file_info.get("file_size", 0)

        try:
            # Download file
            data_url = f"{GDC_API_BASE}/data/{file_id}"
            file_response = requests.get(data_url, timeout=30)

            if file_response.status_code == 200:
                file_path = output_dir / file_name
                with open(file_path, "wb") as f:
                    f.write(file_response.content)
                downloaded_count += 1
                print(f"Downloaded: {file_name} ({file_size / (1024**2):.2f} MB)")
            else:
                failed_count += 1
                print(f"Failed to download: {file_name} (status: {file_response.status_code})")
        except Exception as e:
            failed_count += 1
            print(f"Error downloading {file_name}: {e}")

    # Save manifest
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "description": "GDC Breast Cancer Genomic Data",
            "files_downloaded": downloaded_count,
            "files_failed": failed_count,
            "total_size_gb": current_size / (1024**3),
            "files": selected_files
        }, f, indent=2)

    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Files saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/gdc_data")
    download_gdc_genomic_data(output_dir, target_size_gb=2.0)  # Start with 2 GB for testing
