"""Main entry point for breast cancer AI model."""

import argparse
from pathlib import Path

from .model import BreastCancerClassifier, BreastCancerPredictor
from .data_fetcher import IDCDataFetcher, fetch_advanced_mri_lesions
from .preprocessing import get_preprocessor


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Breast Cancer AI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fetch data command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch data from IDC")
    fetch_parser.add_argument(
        "--collection",
        choices=["advanced_mri_breast_lesions", "breast_cancer_screening_dbt", "cbis_ddsm"],
        default="advanced_mri_breast_lesions",
        help="Collection to fetch",
    )
    fetch_parser.add_argument("--limit", type=int, default=50, help="Number of series to fetch")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict on a scan")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to DICOM image")
    predict_parser.add_argument("--modality", type=str, default="MR", help="Imaging modality")

    # List collections command
    subparsers.add_parser("list", help="List available collections")

    args = parser.parse_args()

    if args.command == "fetch":
        print(f"Fetching data from {args.collection}...")
        if args.collection == "advanced_mri_breast_lesions":
            cohort_info = fetch_advanced_mri_lesions(limit=args.limit)
        else:
            fetcher = IDCDataFetcher()
            cohort_info = fetcher.build_cohort(args.collection, max_series=args.limit)

        print(f"Cohort query: {cohort_info}")
        print("\nTo download, use the NCI IDC MCP tools with this query.")

    elif args.command == "predict":
        print(f"Loading model from {args.model}...")
        predictor = BreastCancerPredictor.load(args.model)

        print(f"Preprocessing image: {args.image}")
        preprocessor = get_preprocessor(args.modality)
        image_tensor = preprocessor(args.image)

        print("Making prediction...")
        result = predictor.predict(image_tensor.unsqueeze(0))

        print("\n" + "="*50)
        print("BREAST CANCER ANALYSIS RESULTS")
        print("="*50)
        print(f"\nCancer Probability: {result['cancer_probability']:.2%}")
        print(f"Classification: {'MALIGNANT' if result['is_malignant'] else 'BENIGN'}")
        print(f"\nGrade: {result['grade']}")
        print("Grade Probabilities:")
        for grade, prob in result['grade_probabilities'].items():
            print(f"  {grade}: {prob:.2%}")

        print(f"\nReceptor Status:")
        for receptor, positive in result['receptor_status'].items():
            status = "POSITIVE" if positive else "NEGATIVE"
            print(f"  {receptor}: {status}")

        print(f"\nTreatment Response: {result['treatment_response']}")
        print("Response Probabilities:")
        for response, prob in result['response_probabilities'].items():
            print(f"  {response}: {prob:.2%}")

    elif args.command == "list":
        fetcher = IDCDataFetcher()
        collections = fetcher.list_breast_collections()

        print("\nAvailable Breast Cancer Collections:")
        print("="*50)
        for col in collections:
            print(f"\nID: {col['id']}")
            print(f"Name: {col['name']}")
            print(f"Modality: {col['modality']}")
            print(f"Labels: {', '.join(col['labels'])}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
