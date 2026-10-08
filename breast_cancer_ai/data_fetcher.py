"""Data fetcher for NCI Imaging Data Commons breast cancer datasets."""

import json
from typing import List, Dict, Optional


class IDCDataFetcher:
    """Fetch breast cancer imaging data from NCI IDC."""

    BREAST_COLLECTIONS = {
        "advanced_mri_breast_lesions": {
            "name": "Advanced-MRI-Breast-Lesions",
            "modality": "MR",
            "labels": ["malignant_benign", "grade", "receptor_status", "ki67"],
        },
        "breast_cancer_screening_dbt": {
            "name": "Breast-Cancer-Screening-DBT",
            "modality": "MG",
            "labels": ["normal", "actionable", "benign", "cancer"],
        },
        "cbis_ddsm": {
            "name": "CBIS-DDSM",
            "modality": "MG",
            "labels": ["normal", "benign", "malignant"],
        },
        "acrin_6698": {
            "name": "ACRIN-6698",
            "modality": "MR",
            "labels": ["treatment_response"],
        },
    }

    def __init__(self):
        """Initialize data fetcher."""
        self.available_servers = []

    def check_mcp_availability(self) -> bool:
        """Check if NCI IDC MCP server is available."""
        try:
            # This would use the MCP tools when called from the AI agent
            return True
        except Exception:
            return False

    def get_collection_info(self, collection_id: str) -> Dict:
        """Get information about a collection.

        Args:
            collection_id: Collection ID from IDC

        Returns:
            Collection metadata
        """
        if collection_id not in self.BREAST_COLLECTIONS:
            raise ValueError(f"Unknown collection: {collection_id}")

        return self.BREAST_COLLECTIONS[collection_id]

    def list_breast_collections(self) -> List[Dict]:
        """List all breast cancer collections.

        Returns:
            List of collection information
        """
        return [
            {"id": cid, **info}
            for cid, info in self.BREAST_COLLECTIONS.items()
        ]

    def build_cohort(
        self,
        collection_id: str,
        modality: Optional[str] = None,
        max_series: int = 100,
    ) -> Dict:
        """Build a cohort query for IDC.

        Args:
            collection_id: Collection ID
            modality: Filter by modality (MR, MG, etc.)
            max_series: Maximum number of series to return

        Returns:
            Cohort query parameters
        """
        collection_info = self.get_collection_info(collection_id)

        # Build terms for cohort
        terms = {
            "collection_id": [collection_id],
        }

        if modality:
            terms["Modality"] = [modality]
        else:
            terms["Modality"] = [collection_info["modality"]]

        return {
            "terms": terms,
            "max_series": max_series,
        }

    def parse_cohort_response(self, response: str) -> Dict:
        """Parse cohort response from IDC.

        Args:
            response: JSON string response from IDC

        Returns:
            Parsed cohort data
        """
        data = json.loads(response)

        return {
            "total_series": data.get("total_series", 0),
            "total_patients": data.get("total_patients", 0),
            "total_size_TB": data.get("total_size_TB", 0),
            "series_sample": data.get("series_sample", []),
        }

    def get_download_urls(self, series_list: List[str]) -> List[str]:
        """Get download URLs for series.

        Args:
            series_list: List of series instance UIDs

        Returns:
            List of download URLs
        """
        # This would call the IDC get_cohort_urls tool
        # For now, return placeholder
        return [
            f"https://idc-data.appspot.com/{series_uid}"
            for series_uid in series_list
        ]


# Example usage functions that would be called by the AI agent
def fetch_advanced_mri_lesions(limit: int = 50) -> Dict:
    """Fetch Advanced-MRI-Breast-Lesions dataset.

    Args:
        limit: Maximum number of series to fetch

    Returns:
        Cohort information
    """
    fetcher = IDCDataFetcher()
    cohort_query = fetcher.build_cohort(
        "advanced_mri_breast_lesions",
        modality="MR",
        max_series=limit,
    )

    # This would be called via MCP tools:
    # result = mcp_call_tool("nci-imaging-data-commons", "build_cohort", cohort_query)

    return {
        "query": cohort_query,
        "collection": fetcher.get_collection_info("advanced_mri_breast_lesions"),
    }


def fetch_breast_screening_dbt(limit: int = 100) -> Dict:
    """Fetch Breast-Cancer-Screening-DBT dataset.

    Args:
        limit: Maximum number of series to fetch

    Returns:
        Cohort information
    """
    fetcher = IDCDataFetcher()
    cohort_query = fetcher.build_cohort(
        "breast_cancer_screening_dbt",
        modality="MG",
        max_series=limit,
    )

    return {
        "query": cohort_query,
        "collection": fetcher.get_collection_info("breast_cancer_screening_dbt"),
    }
