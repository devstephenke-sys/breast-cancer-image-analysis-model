# Breast Cancer Image Analysis AI

AI model for breast cancer image classification and grading using NCI Imaging Data Commons datasets.

## Datasets

### Primary Dataset: Advanced-MRI-Breast-Lesions
- **Collection ID**: `advanced_mri_breast_lesions`
- **Patients**: 632
- **Size**: 0.646 TB
- **Modality**: MRI (T1-weighted DCE, T2-weighted)
- **Labels**: Malignant/Benign, Pathology Grade, Receptor Status, KI67
- **Features**: Treatment Response Assessment Maps (TRAMs)

### Secondary Datasets
- **Breast-Cancer-Screening-DBT**: 5,060 patients, DBT imaging
- **CBIS-DDSM**: 6,671 patients, Mammography
- **ACRIN-6698**: 385 patients, DWI MRI for treatment response

## Model Architecture

Multi-task learning model for:
1. **Binary Classification**: Malignant vs Benign
2. **Grade Classification**: Low, Intermediate, High grade
3. **Receptor Status Prediction**: ER+, PR+, HER2+
4. **Treatment Response**: Complete vs Partial vs No response

## Pipeline

1. **Data Fetching**: Use NCI IDC MCP to download images
2. **Preprocessing**: DICOM normalization, resizing, augmentation
3. **Training**: 3D CNN for MRI, 2D CNN for mammography
4. **Inference**: Accept scan, return cancer probability and grade

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from breast_cancer_ai import BreastCancerClassifier

# Load model
model = BreastCancerClassifier.load('models/best_model.pth')

# Predict on scan
result = model.predict('path/to/scan.dcm')
print(f"Cancer Probability: {result.cancer_probability}")
print(f"Grade: {result.grade}")
print(f"Receptor Status: {result.receptor_status}")
```
