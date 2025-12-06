# 🌿 Plant Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

An AI-powered plant disease detection system for precision agriculture, using deep learning to identify 10 common tomato diseases with 90%+ accuracy.

## 📋 Overview

Plant diseases cause significant crop losses worldwide. This system helps farmers identify diseases early using computer vision and transfer learning, enabling timely intervention and reducing crop damage.

**Key Features:**
- 🤖 EfficientNetB0-based deep learning model
- 🔍 Grad-CAM visualization for explainability
- ⚡ Real-time inference (<200ms)
- 🌐 RESTful API with FastAPI
- 🎨 Clean web interface
- 🐳 Docker deployment ready
- 📊 10 disease categories

## 🎯 Supported Diseases

1. **Bacterial Spot** - Bacterial infection causing dark spots
2. **Early Blight** - Fungal disease with concentric rings
3. **Late Blight** - Devastating fungal disease
4. **Leaf Mold** - Fungal growth on leaves
5. **Septoria Leaf Spot** - Common fungal infection
6. **Spider Mites** - Pest-induced damage
7. **Target Spot** - Fungal disease with target-like patterns
8. **Yellow Leaf Curl Virus** - Viral infection
9. **Mosaic Virus** - Viral leaf discoloration
10. **Healthy** - No disease detected

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 92.5% |
| **Top-3 Accuracy** | 98.1% |
| **Inference Time** | ~180ms |
| **Training Time** | ~25 minutes |
| **Model Size** | ~25MB |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 5GB free disk space
- (Optional) Docker

### Installation

1. **Clone & Setup**
```bash
git clone https://github.com/mahmooooudz/plant-disease-detector.git
cd plant-disease-detector

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

2. **Download Dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset):
- Extract to `data/raw/`
- Should have structure: `data/raw/New Plant Diseases Dataset(Augmented)/train/` and `/valid/`

3. **Train Model**
```bash
python src/train.py
```
Expected output: ~92% validation accuracy in 20-25 minutes

4. **Run API**
```bash
cd api
python app.py
```

5. **Open Browser**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

## 📁 Project Structure

```
plant-disease-detector/
├── data/
│   ├── raw/                    # Dataset
│   ├── processed/              # Auto-generated
│   └── test_samples/           # Test images
├── models/
│   ├── plant_disease_detector.h5  # Trained model
│   ├── model_metrics.json         # Performance metrics
│   ├── training_history.png       # Training curves
│   └── confusion_matrix.png       # Confusion matrix
├── src/
│   ├── data_preprocessing.py      # Data handling
│   ├── model_architecture.py      # Model definition
│   ├── train.py                   # Training script
│   └── inference.py               # Inference engine
├── api/
│   ├── app.py                     # FastAPI backend
│   └── requirements.txt
├── frontend/
│   └── index.html                 # Web interface
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture

**Base Model:** EfficientNetB0 (pre-trained on ImageNet)

**Custom Head:**
- GlobalAveragePooling
- BatchNormalization + Dropout(0.4)
- Dense(256) + ReLU + BatchNorm + Dropout(0.3)
- Dense(128) + ReLU + Dropout(0.2)
- Dense(10) + Softmax

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Augmentation: Rotation, shifts, zoom, flips
- Early Stopping: Patience=5

## 📡 API Endpoints

### POST /predict
Upload leaf image and get disease prediction with Grad-CAM.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_image.jpg" \
  -F "include_gradcam=true"
```

**Response:**
```json
{
  "prediction": "Early blight",
  "confidence": 0.94,
  "top_predictions": [
    {"disease": "Early blight", "confidence": 0.94},
    {"disease": "Late blight", "confidence": 0.03},
    {"disease": "Septoria leaf spot", "confidence": 0.02}
  ],
  "inference_time_ms": 178.3,
  "gradcam_available": true,
  "gradcam_base64": "..."
}
```

### GET /health
Health check endpoint.

### GET /metrics
Get model performance metrics.

## 🔬 How It Works

1. **Image Upload:** User uploads leaf image through web UI
2. **Preprocessing:** Image resized to 224x224, normalized
3. **Inference:** EfficientNetB0 model predicts disease class
4. **Grad-CAM:** Generates heatmap showing focus areas
5. **Results:** Returns top-3 predictions with confidence scores

## 🎨 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which parts of the leaf the AI focuses on:
- Red areas: High importance for prediction
- Blue/green areas: Lower importance
- Helps farmers understand AI reasoning
- Validates model is looking at actual disease symptoms

## 🎯 Engineering Decisions

### Why EfficientNetB0?
- **Accuracy:** Outperforms ResNet50 with fewer parameters
- **Speed:** 180ms inference (production-ready)
- **Size:** 25MB model (edge deployment possible)
- **Proven:** State-of-the-art for image classification

### Why FastAPI?
- **Performance:** Async support for concurrent requests
- **Documentation:** Automatic OpenAPI docs
- **Modern:** Type hints and validation with Pydantic

### Why 10 Classes?
- Balanced complexity vs. real-world utility
- All tomato diseases (consistent plant type)
- Dataset has sufficient samples per class
- Demonstrates multi-class capability

## 📈 Training Process

The training script:
1. Loads balanced dataset (87,000+ images)
2. Applies data augmentation
3. Fine-tunes EfficientNetB0
4. Monitors validation accuracy
5. Saves best model
6. Generates performance visualizations

**Typical Training Output:**
```
Epoch 1/20: val_accuracy: 0.7842
Epoch 5/20: val_accuracy: 0.8923
Epoch 10/20: val_accuracy: 0.9156
Epoch 15/20: val_accuracy: 0.9247
Final: val_accuracy: 0.9251 (92.51%)
```

## 🌟 Use Cases

1. **Precision Agriculture:** Early disease detection
2. **Agricultural Extension:** Remote diagnosis tool
3. **Research:** Disease pattern analysis
4. **Education:** Agricultural training tool
5. **Mobile Apps:** Farmer-facing applications

## 🔮 Future Improvements

- [ ] Expand to 38 plant species (full PlantVillage dataset)
- [ ] Mobile app (React Native + TensorFlow Lite)
- [ ] Disease severity assessment
- [ ] Treatment recommendations
- [ ] Multi-language support
- [ ] Offline mode for remote areas
- [ ] Integration with IoT sensors

## 📊 Dataset

**Source:** PlantVillage Dataset (augmented version)
- **Total Images:** 87,000+
- **Classes:** 10 tomato diseases + healthy
- **Quality:** High-resolution leaf images
- **Balance:** Each class has 8,000-9,000 images

**Citation:**
```
Hughes, D. P., & Salathe, M. (2015).
An open access repository of images on plant health to enable the
development of mobile disease diagnostics.
arXiv preprint arXiv:1511.08060.
```

## 🤝 Contributing

This project was built as a technical demonstration for agricultural AI applications.

## 📝 License

MIT License - see LICENSE file

## 👤 Contact

**Mahmoud Emad**
- Email: Mahmoudkhafaga73@gmail.com
- LinkedIn: [linkedin.com/in/mahmooooudz](https://linkedin.com/in/mahmooooudz/)
- GitHub: [github.com/mahmooooudz](https://github.com/mahmooooudz)

---

**Built with ❤️ for sustainable agriculture and precision farming**

*Empowering farmers with AI to reduce crop losses and increase food security*