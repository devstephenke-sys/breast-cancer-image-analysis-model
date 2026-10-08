"""Flask backend API for Breast Cancer AI GUI."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pydicom
from PIL import Image
import io

# Import the prediction logic
from predict import load_dicom_image, predict, SimpleCNN

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_PATH = Path("data/model.pth")
ALLOWED_EXTENSIONS = {'dcm', 'dicom', 'png', 'jpg', 'jpeg', 'tiff'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the model on startup."""
    global model
    try:
        if MODEL_PATH.exists():
            model = SimpleCNN().to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            # Create a dummy model for demonstration
            model = SimpleCNN().to(device)
            print("Using dummy model for demonstration")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = SimpleCNN().to(device)


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('.', path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """Handle image upload and prediction."""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: .dcm, .dicom, .png, .jpg, .jpeg, .tiff'}), 400

        # Save the file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        # Process the image
        try:
            # Check if it's a DICOM file
            if filename.lower().endswith(('.dcm', '.dicom')):
                # Use existing DICOM loading
                result = predict(model, str(filepath), device)
            else:
                # Convert non-DICOM images to DICOM-like format for prediction
                image = Image.open(filepath).convert('L')  # Convert to grayscale
                image_array = np.array(image).astype(np.float32)
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)

                # Resize to target size
                image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                image_pil = image_pil.resize((256, 256))
                image_array = np.array(image_pil).astype(np.float32) / 255.0

                # Convert to tensor
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)

                # Run prediction
                model.eval()
                with torch.no_grad():
                    outputs = model(image_tensor)

                # Get predictions
                cancer_prob = outputs["cancer"].item()
                grade_idx = outputs["grade"].argmax().item()
                response_idx = outputs["response"].argmax().item()

                grade_labels = ["low", "intermediate", "high"]
                response_labels = ["complete", "partial", "no_response"]

                result = {
                    "cancer_probability": cancer_prob,
                    "grade": grade_labels[grade_idx],
                    "treatment_response": response_labels[response_idx],
                }

            # Clean up uploaded file
            filepath.unlink()

            # Return results
            return jsonify({
                'success': True,
                'predictions': {
                    'cancer_probability': result['cancer_probability'],
                    'is_malignant': result['cancer_probability'] > 0.5,
                    'grade': result['grade'].capitalize(),
                    'treatment_response': result['treatment_response'].replace('_', ' ').capitalize()
                },
                'confidence': {
                    'malignant': result['cancer_probability'] if result['cancer_probability'] > 0.5 else 1 - result['cancer_probability'],
                    'benign': 1 - result['cancer_probability'] if result['cancer_probability'] > 0.5 else result['cancer_probability']
                }
            })

        except Exception as e:
            # Clean up file on error
            if filepath.exists():
                filepath.unlink()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/info', methods=['GET'])
def model_info():
    """Return model information."""
    return jsonify({
        'model_type': 'Breast Cancer Classifier',
        'version': '1.0.0',
        'supported_formats': ['DICOM (.dcm, .dicom)', 'Images (.png, .jpg, .jpeg, .tiff)'],
        'predictions': [
            'Cancer Probability (Malignant vs Benign)',
            'Grade (Low, Intermediate, High)',
            'Treatment Response (Complete, Partial, No Response)'
        ],
        'device': device
    })


if __name__ == '__main__':
    # Load model on startup
    load_model()
    print(f"Starting Flask server on http://localhost:5000")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model path: {MODEL_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5000)
