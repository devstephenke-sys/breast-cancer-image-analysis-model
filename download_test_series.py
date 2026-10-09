"""Download a test series from IDC to verify download works."""

import idc_index
from pathlib import Path

def download_test_series():
    """Download one patient's data as a test."""
    output_dir = Path("C:/Users/DBTECH AFRICA/Desktop/Cancer Research/breast-cancer-ai/data/idc_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading test series (AMBL-608)...")
    
    # Download patient AMBL-608 (largest single patient ~1.4 GB)
    client = idc_index.IDCClient()
    
    series_uids = [
        "1.3.6.1.4.1.14519.5.2.1.100375458753755104578276100137784934676",
        "1.3.6.1.4.1.14519.5.2.1.143801325231208788666726450418062179872",
    ]
    
    try:
        for series_uid in series_uids:
            print(f"Downloading series: {series_uid}")
            client.download_dicom_series(
                seriesInstanceUID=series_uid,
                downloadDir=str(output_dir),
                quiet=False,
                show_progress_bar=True
            )
            print(f"Completed series: {series_uid}")
        
        print(f"\nDownload complete! Data saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_test_series()
