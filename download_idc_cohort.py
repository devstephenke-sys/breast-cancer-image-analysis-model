"""Download IDC breast MRI cohort for training (target ~2-3 GB) using idc-index with SQL query."""

import idc_index
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def download_series():
    """Download breast MRI series using idc-index."""
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/idc_cohort_2gb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Querying IDC for breast MRI series...")

    # Create the index client
    client = idc_index.IDCClient()

    # Query for breast MRI series from ACRIN-6698 (smaller cohort ~2-3 GB)
    sql_query = """
    SELECT 
        SeriesInstanceUID,
        PatientID,
        series_size_MB,
        crdc_series_uuid
    FROM index
    WHERE collection_id = 'acrin_6698'
    AND Modality = 'MR'
    ORDER BY series_size_MB DESC
    LIMIT 3
    """

    df = client.sql_query(sql_query)
    
    print(f"Found {len(df)} series")
    print(f"Total size: {df['series_size_MB'].sum() / 1024:.2f} GB")
    print("\nSeries to download:")
    print(df[['PatientID', 'series_size_MB']].to_string())
    
    # Get the series UIDs
    series_uids = df['SeriesInstanceUID'].tolist()
    
    print(f"\nStarting download of {len(series_uids)} series...")
    print("This may take a while for 2-3 GB of data...")
    
    # Download each series one by one
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
    download_series()
