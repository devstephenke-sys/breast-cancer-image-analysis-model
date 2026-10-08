"""Download ACRIN-6698 test sample (10 patients with complete clinical data)."""

import idc_index
from pathlib import Path
from tqdm import tqdm

def download_acrin_test():
    """Download 10 patients from ACRIN-6698 with complete clinical data."""
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/idc-data/acrin_6698_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 10 patients with complete clinical data (ltype, hrher4g, sbrgrade)
    patient_ids = [
        "ACRIN-6698-453236",
        "ACRIN-6698-415631",
        "ACRIN-6698-638126",
        "ACRIN-6698-579123",
        "ACRIN-6698-475650",
        "ACRIN-6698-345907",
        "ACRIN-6698-653262",
        "ACRIN-6698-258713",
        "ACRIN-6698-710197",
        "ACRIN-6698-563681",
    ]

    print(f"Downloading {len(patient_ids)} patients from ACRIN-6698...")
    print(f"Output directory: {output_dir}")

    client = idc_index.IDCClient()

    # Query for series for these patients
    patient_list = "', '".join(patient_ids)
    sql_query = f"""
    SELECT SeriesInstanceUID, PatientID, series_size_MB
    FROM index
    WHERE collection_id = 'acrin_6698'
    AND Modality = 'MR'
    AND PatientID IN ('{patient_list}')
    ORDER BY PatientID
    """

    df = client.sql_query(sql_query)
    print(f"Found {len(df)} series")
    print(f"Total size: {df['series_size_MB'].sum() / 1024:.2f} GB")

    # Download series
    series_uids = df['SeriesInstanceUID'].tolist()
    print(f"\nStarting download of {len(series_uids)} series...")

    downloaded = 0
    failed = 0

    for series_uid in tqdm(series_uids, desc="Downloading series"):
        try:
            client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(output_dir),
                quiet=False,
                show_progress_bar=True
            )
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {series_uid}: {e}")
            failed += 1

    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {downloaded} series")
    print(f"Failed: {failed} series")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    download_acrin_test()
