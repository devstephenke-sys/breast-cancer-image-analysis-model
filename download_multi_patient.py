"""Download multiple patients from IDC to reach ~2-3 GB total."""

import idc_index
from pathlib import Path
from tqdm import tqdm

def download_multiple_patients():
    """Download multiple patients to reach ~2-3 GB."""
    output_dir = Path("C:/Users\DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/idc_cohort_2gb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading patients from Advanced-MRI-Breast-Lesions...")
    
    # Get series UIDs for top 5 patients (approximately 2-3 GB)
    client = idc_index.IDCClient()
    
    # Top 5 patients by size from the query earlier
    patients = [
        "AMBL-608",  # ~1.44 GB
        "AMBL-559",  # ~1.44 GB
        "AMBL-578",  # ~1.43 GB
        "AMBL-563",  # ~1.43 GB
        "AMBL-610",  # ~1.32 GB
    ]
    
    # Get all series for these patients
    sql_query = """
    SELECT SeriesInstanceUID
    FROM index
    WHERE collection_id = 'advanced_mri_breast_lesions'
    AND Modality = 'MR'
    AND PatientID IN ('AMBL-608', 'AMBL-559', 'AMBL-578', 'AMBL-563', 'AMBL-610')
    """
    
    df = client.sql_query(sql_query)
    series_uids = df['SeriesInstanceUID'].tolist()
    
    print(f"Found {len(series_uids)} series across {len(patients)} patients")
    print(f"Estimated total size: ~6-7 GB (will download 2-3 GB subset)")
    
    # Download first 3 patients (AMBL-608, AMBL-559, AMBL-578) for ~4.3 GB
    target_patients = ['AMBL-608', 'AMBL-559', 'AMBL-578']
    
    sql_query_subset = """
    SELECT SeriesInstanceUID
    FROM index
    WHERE collection_id = 'advanced_mri_breast_lesions'
    AND Modality = 'MR'
    AND PatientID IN ('AMBL-608', 'AMBL-559', 'AMBL-578')
    """
    
    df_subset = client.sql_query(sql_query_subset)
    series_uids_subset = df_subset['SeriesInstanceUID'].tolist()
    
    print(f"Downloading {len(series_uids_subset)} series for 3 patients (~4.3 GB)...")
    
    downloaded = 0
    failed = 0
    
    for series_uid in tqdm(series_uids_subset, desc="Downloading series"):
        try:
            client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(output_dir),
                quiet=True,
                show_progress_bar=False
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
    download_multiple_patients()
