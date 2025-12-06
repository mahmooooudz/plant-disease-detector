"""
Inference engine with Grad-CAM for plant disease detection
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import time


class PlantDiseaseDetector:
    def __init__(self, model_path='models/plant_disease_detector.h5'):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.img_size = (224, 224)
        
        # Load model
        self.load_model()
        
        # Load class names
        self.load_class_names()
        
    def load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading model from {self.model_path}...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
        
        # Warm up
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        _ = self.model.predict(dummy, verbose=0)
        print("🔥 Model ready!")
        
    def load_class_names(self):
        """Load class names from metrics"""
        metrics_path = self.model_path.parent / 'model_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.class_names = metrics['class_names']
        else:
            # Default fallback
            self.class_names = [
                'Tomato___Bacterial_spot',
                'Tomato___Early_blight',
                'Tomato___Late_blight',
                'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites',
                'Tomato___Target_Spot',
                'Tomato___Yellow_Leaf_Curl_Virus',
                'Tomato___mosaic_virus',
                'Tomato___healthy'
            ]
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image or numpy array
            
        Returns:
            Preprocessed image batch and original image
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            img = image_path
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Invalid image input")
        
        original_img = img.copy()
        
        # Resize and normalize
        img_resized = cv2.resize(img, self.img_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, original_img
    
    def predict(self, image_path, return_gradcam=True, top_k=3):
        """
        Make prediction
        
        Args:
            image_path: Path to image
            return_gradcam: Whether to generate Grad-CAM
            top_k: Return top-k predictions
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        img_batch, original_img = self.preprocess_image(image_path)
        preprocess_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(img_batch, verbose=0)[0]
        inference_time = time.time() - start_time
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            disease_name = self.class_names[idx].replace('Tomato___', '').replace('_', ' ')
            confidence = float(predictions[idx])
            top_predictions.append({
                'disease': disease_name,
                'confidence': confidence
            })
        
        result = {
            'prediction': top_predictions[0]['disease'],
            'confidence': top_predictions[0]['confidence'],
            'top_predictions': top_predictions,
            'inference_time_ms': inference_time * 1000,
            'preprocess_time_ms': preprocess_time * 1000,
            'total_time_ms': (inference_time + preprocess_time) * 1000
        }
        
        # Generate Grad-CAM
        if return_gradcam:
            gradcam_start = time.time()
            gradcam_img = self.generate_gradcam(img_batch, original_img, top_indices[0])
            result['gradcam_time_ms'] = (time.time() - gradcam_start) * 1000
            result['gradcam_image'] = gradcam_img
        
        return result
    
    def generate_gradcam(self, img_batch, original_img, pred_index):
        """
        Generate Grad-CAM visualization
        
        Args:
            img_batch: Preprocessed image
            original_img: Original image
            pred_index: Predicted class index
            
        Returns:
            Grad-CAM overlay image
        """
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return original_img
        
        # Create gradient model
        grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            loss = predictions[:, pred_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original to BGR
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Overlay
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return overlay_rgb
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_path': str(self.model_path),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape
        }


if __name__ == "__main__":
    print("\n🌿 PLANT DISEASE DETECTOR - INFERENCE TEST\n")
    
    detector = PlantDiseaseDetector()
    info = detector.get_model_info()
    
    print(f"📊 Model loaded")
    print(f"   Classes: {info['num_classes']}")
    print("\n✅ Ready for inference!")