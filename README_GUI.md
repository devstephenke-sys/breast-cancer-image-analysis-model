# Breast Cancer AI - GUI Application

A modern, fancy web-based GUI for breast cancer medical imaging analysis using AI.

## Features

- **Modern UI**: Beautiful gradient-based design with smooth animations
- **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- **Multiple Format Support**: Accepts DICOM (.dcm, .dicom) and standard image formats (.png, .jpg, .jpeg, .tiff)
- **Real-time Analysis**: AI-powered predictions for cancer detection
- **Comprehensive Results**: Displays cancer probability, grade classification, and treatment response
- **Responsive Design**: Works on desktop and mobile devices
- **Medical Disclaimer**: Includes appropriate medical warnings

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open in Browser

Navigate to `http://localhost:5000` in your web browser

## Usage

1. **Upload Image**: Drag and drop a medical image or click to browse
2. **Select File**: Choose a DICOM file or standard image format
3. **Analyze**: Click the "Analyze Image" button
4. **View Results**: See the AI predictions including:
   - Cancer probability (Malignant vs Benign)
   - Tumor grade (Low, Intermediate, High)
   - Treatment response prediction

## API Endpoints

### `GET /`
- Returns the main HTML page

### `GET /api/health`
- Health check endpoint
- Returns server status and model information

### `POST /api/predict`
- Upload image and get predictions
- Accepts multipart/form-data with file field
- Returns JSON with predictions and confidence scores

### `GET /api/info`
- Returns model information and supported formats

## File Structure

```
breast-cancer-ai/
├── app.py              # Flask backend server
├── index.html          # Main HTML page
├── styles.css          # Modern CSS styling
├── script.js           # JavaScript functionality
├── predict.py          # Prediction logic
├── model.py            # Model architecture
├── preprocessing.py    # Image preprocessing
├── requirements.txt    # Python dependencies
├── uploads/           # Temporary upload directory
└── data/
    └── model.pth      # Trained model weights
```

## Model Information

The GUI uses a multi-task breast cancer classification model that predicts:

1. **Binary Classification**: Malignant vs Benign
2. **Grade Classification**: Low, Intermediate, High
3. **Receptor Status**: ER, PR, HER2
4. **Treatment Response**: Complete, Partial, No Response

## Technical Stack

- **Backend**: Flask with Flask-CORS
- **Frontend**: Vanilla HTML, CSS, JavaScript
- **AI Framework**: PyTorch
- **Medical Imaging**: pydicom, MONAI
- **Styling**: Custom CSS with gradients and animations

## Important Notes

- This is a demonstration/prototype tool
- Results should be reviewed by qualified medical professionals
- Not intended for diagnostic use without clinical validation
- Maximum file size: 16MB
- Supports DICOM and standard image formats

## Customization

### Change Port

Edit `app.py` and modify the port in the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to desired port
```

### Change Styling

Edit `styles.css` to customize colors, gradients, and animations. The CSS uses CSS variables for easy theming:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    /* ... more variables */
}
```

### Modify Model

Update the model path in `app.py`:
```python
MODEL_PATH = Path("data/model.pth")  # Change to your model path
```

## Troubleshooting

### Model Not Found
If you see "Model file not found", ensure:
- The model file exists at `data/model.pth`
- Or update the `MODEL_PATH` in `app.py`

### File Upload Issues
- Check file size (max 16MB)
- Verify file format (DICOM or image)
- Ensure uploads directory exists

### Server Won't Start
- Check if port 5000 is already in use
- Verify Flask is installed: `pip install flask flask-cors`
- Check Python version (requires Python 3.7+)

## License

This project is for research and educational purposes.

## Credits

Built with PyTorch, Flask, and modern web technologies.
