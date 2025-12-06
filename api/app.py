"""
FastAPI backend for plant disease detection
"""
import sys
sys.path.append('..')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import base64
import time
from pathlib import Path
import json

from src.inference import PlantDiseaseDetector

# Initialize app
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_path = Path("../frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global detector
detector = None

def get_detector():
    """Get or initialize detector"""
    global detector
    if detector is None:
        print("🔄 Loading model...")
        detector = PlantDiseaseDetector(model_path='models/plant_disease_detector.h5')
        print("✅ Model ready!")
    return detector


# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: list
    inference_time_ms: float
    total_time_ms: float
    gradcam_available: bool
    gradcam_base64: str = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    val_accuracy: float
    num_classes: int
    training_date: str


# Routes
@app.get("/", response_class=FileResponse)
async def root():
    """Serve frontend"""
    frontend_file = frontend_path / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    return {"message": "Plant Disease Detection API is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model metrics"""
    try:
        det = get_detector()
        metrics_path = Path('../models/model_metrics.json')
        
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Metrics not found")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return MetricsResponse(
            val_accuracy=metrics.get('val_accuracy', 0.0),
            num_classes=metrics.get('num_classes', 10),
            training_date=metrics.get('training_date', 'Unknown')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = True
):
    """
    Predict plant disease from leaf image
    
    Args:
        file: Uploaded image
        include_gradcam: Include Grad-CAM visualization
        
    Returns:
        Prediction results
    """
    start_time = time.time()
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Convert to numpy
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert to RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Get detector
        det = get_detector()
        
        # Predict
        result = det.predict(image_np, return_gradcam=include_gradcam, top_k=3)
        
        # Encode Grad-CAM
        gradcam_base64 = None
        if 'gradcam_image' in result:
            gradcam_pil = Image.fromarray(result['gradcam_image'])
            buffer = io.BytesIO()
            gradcam_pil.save(buffer, format='JPEG', quality=95)
            gradcam_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        total_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            top_predictions=result['top_predictions'],
            inference_time_ms=result['inference_time_ms'],
            total_time_ms=total_time,
            gradcam_available=include_gradcam,
            gradcam_base64=gradcam_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "="*60)
    print("🚀 STARTING PLANT DISEASE DETECTION API")
    print("="*60)
    get_detector()
    print("\n✅ API ready!")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)