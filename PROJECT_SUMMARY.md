# Breast Cancer AI Project - Complete Setup

## Overview

I've built a complete breast cancer image analysis AI system that can:
1. **Fetch** breast cancer imaging data from NCI Imaging Data Commons
2. **Preprocess** DICOM images (MRI and mammography)
3. **Train** a multi-task deep learning model
4. **Predict** cancer probability, grade, receptor status, and treatment response

## Project Structure

```
breast-cancer-ai/
├── README.md                          # Project documentation
├── config.json                        # Configuration file
├── pyproject.toml                     # Python dependencies
└── breast_cancer_ai/
    ├── __init__.py
    ├── model.py                       # Multi-task CNN architecture
    ├── data_fetcher.py                # NCI IDC data fetching
    ├── preprocessing.py               # DICOM preprocessing pipeline
    ├── train.py                       # Training logic
    └── main.py                        # CLI entry point
```

## Available Breast Cancer Datasets

### 1. Advanced-MRI-Breast-Lesions (RECOMMENDED)
- **632 patients, 0.646 TB**
- **Modality**: MRI (T1-weighted DCE, T2-weighted)
- **Labels**: Malignant/Benign, Pathology Grade, Receptor Status (ER/PR/HER2), KI67
- **Special Features**: Treatment Response Assessment Maps (TRAMs)
- **Best for**: Multi-grade analysis, receptor prediction

### 2. Breast-Cancer-Screening-DBT
- **5,060 patients, 1.638 TB**
- **Modality**: Digital Breast Tomosynthesis (DBT)
- **Labels**: Normal, Actionable, Benign, Cancer with annotations
- **Best for**: Large-scale screening classification

### 3. CBIS-DDSM
- **6,671 patients, 0.164 TB**
- **Modality**: Mammography
- **Labels**: Normal, Benign, Malignant with verified pathology
- **Best for**: Classic mammography classification

### 4. ACRIN-6698
- **385 patients, 0.842 TB**
- **Modality**: Diffusion Weighted Imaging (DWI) MRI
- **Labels**: Neoadjuvant chemotherapy response
- **Best for**: Treatment response prediction

## Model Architecture

### Multi-Task Learning Model

The model performs 4 tasks simultaneously:

1. **Binary Classification**: Malignant vs Benign
2. **Grade Classification**: Low, Intermediate, High grade
3. **Receptor Status**: ER+, PR+, HER2+ (each binary)
4. **Treatment Response**: Complete, Partial, No response

### Architecture Options

- **3D DenseNet121**: For MRI volumes (256x256x32)
- **2D ResNet18**: For mammography (512x512)

## How to Use

### Step 1: Install Dependencies

```bash
cd "C:\Users\DBTECH AFRICA\Desktop\Cancer Research\breast-cancer-ai"
uv sync
```

### Step 2: List Available Collections

```bash
uv run python -m breast_cancer_ai.main list
```

### Step 3: Fetch Data from NCI IDC

Using the MCP tools (I can help you run this):

```python
# This would be called via NCI IDC MCP
from breast_cancer_ai.data_fetcher import fetch_advanced_mri_lesions

cohort_info = fetch_advanced_mri_lesions(limit=50)
```

Then use the cohort query with NCI IDC MCP to download:
```python
# Build cohort
terms = {
    "collection_id": ["advanced_mri_breast_lesions"],
    "Modality": ["MR"]
}

# Call via MCP
result = mcp_call_tool("nci-imaging-data-commons", "build_cohort", {
    "terms": terms,
    "max_series": 50
})
```

### Step 4: Train the Model

Once you have data downloaded:

```python
from breast_cancer_ai.train import BreastCancerTrainer
from breast_cancer_ai.model import BreastCancerClassifier

# Initialize model
model = BreastCancerClassifier(
    spatial_dims=3,  # 3 for MRI, 2 for mammography
    num_classes_grade=3,
    num_classes_response=3,
)

# Initialize trainer
trainer = BreastCancerTrainer(model, device="cuda")

# Train (requires DataLoader with your data)
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Step 5: Predict on New Scans

```bash
uv run python -m breast_cancer_ai.main predict \
    --model checkpoints/best_model.pth \
    --image path/to/scan.dcm \
    --modality MR
```

Or programmatically:

```python
from breast_cancer_ai.model import BreastCancerPredictor
from breast_cancer_ai.preprocessing import get_preprocessor

# Load model
predictor = BreastCancerPredictor.load("checkpoints/best_model.pth")

# Preprocess image
preprocessor = get_preprocessor("MR")
image_tensor = preprocessor("path/to/scan.dcm")

# Predict
result = predictor.predict(image_tensor.unsqueeze(0))

print(f"Cancer Probability: {result['cancer_probability']:.2%}")
print(f"Grade: {result['grade']}")
print(f"Receptor Status: {result['receptor_status']}")
```

## Next Steps

### Immediate (Using MCP Tools):

1. **Build a cohort** from Advanced-MRI-Breast-Lesions:
   - Use NCI IDC MCP `build_cohort` tool
   - Filter for MR modality
   - Set max_series to a manageable number (e.g., 50-100)

2. **Download the data**:
   - Use `get_cohort_urls` to get download URLs
   - Use the returned `idc` commands to download

3. **Extract clinical labels**:
   - Use `list_clinical_tables` to find available clinical data
   - Use `get_clinical_table` to extract grade, receptor status, etc.

### Medium-term:

1. **Create a dataset class** that pairs images with labels
2. **Implement data augmentation** for better generalization
3. **Train the model** on the downloaded data
4. **Evaluate performance** on validation set

### Long-term:

1. **Fine-tune on multiple datasets** (MRI + mammography)
2. **Deploy as a web service** for clinical use
3. **Integrate with hospital PACS** systems
4. **Conduct clinical validation studies**

## Configuration

Edit `config.json` to customize:
- Model architecture (DenseNet vs ResNet)
- Training hyperparameters
- Data paths
- Task weights

## Integration with Your MCP Stack

This AI model integrates with your existing cancer research MCP configuration:

- **BioMCP**: For literature on breast cancer imaging
- **NCI IDC**: For fetching imaging data
- **GDC MCP**: For genomic correlations (e.g., linking imaging to TCGA mutations)
- **Memory**: For storing research hypotheses and results

## Key Features

✅ Multi-task learning (single model for multiple predictions)
✅ Supports both 3D MRI and 2D mammography
✅ Handles DICOM natively
✅ Integrated with NCI IDC via MCP
✅ Configurable architecture
✅ Ready for training and inference

## Data Privacy Note

- All datasets are de-identified
- NCI IDC data is publicly available under CC licenses
- For clinical use, ensure compliance with local regulations (HIPAA, GDPR, etc.)
