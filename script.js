// JavaScript for Breast Cancer AI GUI

class BreastCancerAI {
    constructor() {
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.uploadContainer = document.getElementById('uploadContainer');
        this.filePreview = document.getElementById('filePreview');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.analysisScreen = document.getElementById('analysisScreen');
        this.pixelCanvas = document.getElementById('pixelCanvas');
        this.selectedFile = null;
        this.pixelationAnimation = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkServerHealth();
    }

    setupEventListeners() {
        // Click to upload
        this.uploadZone.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Drag and drop events
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });

        this.uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');

            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Remove file
        document.getElementById('removeBtn').addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeFile();
        });

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => {
            this.analyzeImage();
        });
    }

    async checkServerHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            console.log('Server health:', data);
        } catch (error) {
            console.error('Server health check failed:', error);
        }
    }

    handleFileSelect(file) {
        // Validate file type
        const validExtensions = ['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tiff'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(fileExtension)) {
            this.showError('Invalid file type. Please upload a DICOM or image file.');
            return;
        }

        // Validate file size (max 16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File too large. Maximum size is 16MB.');
            return;
        }

        this.selectedFile = file;
        this.showFilePreview(file);
        this.analyzeBtn.disabled = false;
    }

    showFilePreview(file) {
        // Hide upload zone, show preview
        this.uploadZone.style.display = 'none';
        this.filePreview.style.display = 'flex';

        // Update file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);

        // Update preview icon
        const previewImage = document.getElementById('previewImage');
        const extension = file.name.split('.').pop().toLowerCase();

        if (extension === 'dcm' || extension === 'dicom') {
            previewImage.innerHTML = 'DCM';
        } else {
            previewImage.innerHTML = 'IMG';
        }
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = '';

        // Show upload zone, hide preview
        this.uploadZone.style.display = 'block';
        this.filePreview.style.display = 'none';

        // Disable analyze button
        this.analyzeBtn.disabled = true;

        // Hide results and analysis screen
        this.resultsSection.style.display = 'none';
        this.analysisScreen.style.display = 'none';
        this.stopPixelationAnimation();
    }

    async analyzeImage() {
        if (!this.selectedFile) {
            this.showError('Please select a file first.');
            return;
        }

        // Show analysis screen with pixelation effect
        this.analysisScreen.style.display = 'flex';
        this.startPixelationAnimation();

        // Create form data
        const formData = new FormData();
        formData.append('file', this.selectedFile);

        try {
            // Step 1: Loading Image
            await this.updateStep(1, 'Processing...');
            await this.delay(800);
            await this.updateStep(1, 'Complete', true);

            // Step 2: Pixelation Analysis
            await this.updateStep(2, 'Processing...');
            await this.delay(1000);
            await this.updateStep(2, 'Complete', true);

            // Step 3: Neural Network
            await this.updateStep(3, 'Processing...');
            await this.delay(800);
            await this.updateStep(3, 'Complete', true);

            // Step 4: Generating Results (and actual API call)
            await this.updateStep(4, 'Processing...');

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            await this.updateStep(4, 'Complete', true);

            if (data.success) {
                // Hide analysis screen
                this.stopPixelationAnimation();
                this.analysisScreen.style.display = 'none';
                this.displayResults(data.predictions, data.confidence);
            } else {
                this.stopPixelationAnimation();
                this.analysisScreen.style.display = 'none';
                this.showError(data.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.stopPixelationAnimation();
            this.analysisScreen.style.display = 'none';
            this.showError('Network error. Please check your connection and try again.');
        }
    }

    async updateStep(stepNum, status, completed = false) {
        const step = document.getElementById(`step${stepNum}`);
        const stepStatus = step.querySelector('.step-status');

        step.classList.remove('active', 'completed');
        if (completed) {
            step.classList.add('completed');
            stepStatus.textContent = status;
        } else {
            step.classList.add('active');
            stepStatus.textContent = status;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    startPixelationAnimation() {
        const canvas = this.pixelCanvas;
        const ctx = canvas.getContext('2d');
        const pixelCountEl = document.getElementById('pixelCount');
        const layerNumEl = document.getElementById('layerNum');
        const scanStatusEl = document.getElementById('scanStatus');

        let pixelSize = 50;
        let frame = 0;
        let pixels = 0;
        let layer = 0;

        const animate = () => {
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Create pixelated grid
            const cols = Math.ceil(canvas.width / pixelSize);
            const rows = Math.ceil(canvas.height / pixelSize);

            for (let i = 0; i < cols; i++) {
                for (let j = 0; j < rows; j++) {
                    const x = i * pixelSize;
                    const y = j * pixelSize;

                    // Create gradient based on position and time
                    const hue = (frame + i * 10 + j * 10) % 360;
                    const saturation = 70 + Math.sin(frame * 0.1 + i * 0.5) * 20;
                    const lightness = 30 + Math.cos(frame * 0.1 + j * 0.5) * 20;

                    ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                    ctx.fillRect(x, y, pixelSize - 1, pixelSize - 1);

                    pixels++;
                }
            }

            // Update data display
            pixelCountEl.textContent = pixels.toLocaleString();
            layerNumEl.textContent = layer;

            // Scan status animation
            const statuses = ['SCANNING', 'ANALYZING', 'PROCESSING', 'DETECTING'];
            scanStatusEl.textContent = statuses[Math.floor(frame / 30) % statuses.length];

            // Gradually reduce pixel size for more detail
            if (frame % 60 === 0 && pixelSize > 5) {
                pixelSize -= 5;
                layer++;
            }

            frame++;
            this.pixelationAnimation = requestAnimationFrame(animate);
        };

        animate();
    }

    stopPixelationAnimation() {
        if (this.pixelationAnimation) {
            cancelAnimationFrame(this.pixelationAnimation);
            this.pixelationAnimation = null;
        }
    }

    displayResults(predictions, confidence) {
        // Update cancer probability
        const cancerProb = (predictions.cancer_probability * 100).toFixed(1);
        document.getElementById('cancerProbability').textContent = cancerProb + '%';
        document.getElementById('cancerBar').style.width = cancerProb + '%';

        // Update badge
        const badge = document.getElementById('cancerBadge');
        badge.textContent = predictions.is_malignant ? 'MALIGNANT' : 'BENIGN';
        badge.className = 'result-badge ' + (predictions.is_malignant ? 'malignant' : 'benign');

        // Update confidence values
        document.getElementById('malignantConf').textContent = (confidence.malignant * 100).toFixed(1) + '%';
        document.getElementById('benignConf').textContent = (confidence.benign * 100).toFixed(1) + '%';

        // Update grade
        document.getElementById('gradeValue').textContent = predictions.grade;

        // Update treatment response
        document.getElementById('responseValue').textContent = predictions.treatment_response;

        // Show results section
        this.resultsSection.style.display = 'block';

        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    showError(message) {
        // Create a temporary error message
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1001;
            animation: slideIn 0.3s ease-out;
        `;
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);

        // Remove after 3 seconds
        setTimeout(() => {
            errorDiv.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                document.body.removeChild(errorDiv);
            }, 300);
        }, 3000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new BreastCancerAI();
});
