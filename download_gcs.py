"""Download patients from IDC using GCS bucket instead of AWS."""

import idc_index
from pathlib import Path
from tqdm import tqdm

def download_from_gcs():
    """Download patients using GCS bucket (more reliable)."""
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/idc_cohort_2gb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading patients from Advanced-MRI-Breast-Lesions using GCS...")
    
    client = idc_index.IDCClient()
    
    # Get series for patients we want (AMBL-608, AMBL-559, AMBL-578)
    sql_query = """
    SELECT SeriesInstanceUID
    FROM index
    WHERE collection_id = 'advanced_mri_breast_lesions'
    AND Modality = 'MR'
    AND PatientID IN ('AMBL-608', 'AMBL-559', 'AMBL-578')
    """
    
    df = client.sql_query(sql_query)
    series_uids = df['SeriesInstanceUID'].tolist()
    
    print(f"Found {len(series_uids)} series for 3 patients")
    
    downloaded = 0
    failed = 0
    
    for series_uid in tqdm(series_uids, desc="Downloading series from GCS"):
        try:
            client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(output_dir),
                quiet=True,
                show_progress_bar=False,
                source_bucket_location='gcs'  # Use GCS instead of AWS
            )
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {series_uid}: {e}")
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {downloaded} series")
    print(f"Failed: {failed} series")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    download_from_gcs()
