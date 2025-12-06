"""
Training script for plant disease detection
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model_architecture import create_model, compile_model, get_callbacks
from data_preprocessing import PlantDataPreprocessor


class PlantModelTrainer:
    def __init__(self, data_path, model_save_dir='models'):
        """
        Initialize trainer
        
        Args:
            data_path: Path to dataset
            model_save_dir: Directory to save models
        """
        self.data_path = Path(data_path)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_path = self.data_path / 'train'
        self.val_path = self.data_path / 'valid'
        
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data(self, batch_size=32):
        """Prepare data generators"""
        print("\n" + "="*60)
        print("📊 PREPARING DATA")
        print("="*60)
        
        preprocessor = PlantDataPreprocessor(
            raw_data_path=self.data_path,
            processed_data_path='data/processed'
        )
        
        # Get class info
        class_info = preprocessor.get_class_info()
        self.class_names = class_info['class_names']
        self.num_classes = class_info['num_classes']
        
        print(f"\n📋 Number of classes: {self.num_classes}")
        print(f"📋 Classes: {', '.join(self.class_names[:3])}... (and {self.num_classes - 3} more)")
        
        # Create generators
        self.train_generator, self.val_generator = preprocessor.create_data_generators(
            self.train_path,
            self.val_path,
            batch_size=batch_size
        )
        
        print(f"\n✅ Data prepared!")
        print(f"   Train samples: {self.train_generator.samples}")
        print(f"   Val samples: {self.val_generator.samples}")
        
    def build_model(self):
        """Build and compile model"""
        print("\n" + "="*60)
        print("🏗️ BUILDING MODEL")
        print("="*60)
        
        self.model, self.base_model = create_model(
            num_classes=self.num_classes,
            input_shape=(224, 224, 3)
        )
        
        self.model = compile_model(self.model, learning_rate=0.001)
        
        print("\n✅ Model built successfully!")
        print(f"📊 Total parameters: {self.model.count_params():,}")
        
    def train(self, epochs=20):
        """Train the model"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        model_path = self.model_save_dir / 'plant_disease_detector.h5'
        callbacks = get_callbacks(str(model_path))
        
        start_time = time.time()
        
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {self.training_time:.2f} seconds ({self.training_time/60:.2f} minutes)")
        
    def evaluate(self):
        """Evaluate model"""
        print("\n" + "="*60)
        print("📊 EVALUATING MODEL")
        print("="*60)
        
        # Evaluate
        val_loss, val_acc, val_top3 = self.model.evaluate(self.val_generator, verbose=0)
        
        print(f"\n📈 Validation Performance:")
        print(f"   Accuracy:      {val_acc*100:.2f}%")
        print(f"   Top-3 Accuracy: {val_top3*100:.2f}%")
        print(f"   Loss:          {val_loss:.4f}")
        
        # Get predictions
        self.val_generator.reset()
        y_true = self.val_generator.classes
        y_pred_probs = self.model.predict(self.val_generator, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        print("\n📋 Classification Report:")
        # Get only the classes that actually exist in validation set
        unique_classes = np.unique(y_true)
        actual_class_names = [self.class_names[i] for i in unique_classes]

        report = classification_report(
            y_true, y_pred,
            labels=unique_classes,
            target_names=actual_class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save metrics
        metrics = {
            'val_accuracy': float(val_acc),
            'val_top3_accuracy': float(val_top3),
            'val_loss': float(val_loss),
            'training_time_seconds': self.training_time,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_trained': len(self.history.history['loss'])
        }
        
        metrics_file = self.model_save_dir / 'model_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Metrics saved to {metrics_file}")
        
        # Plot
        self.plot_training_history()
        self.plot_confusion_matrix(cm)
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_history.png', dpi=300)
        print("📊 Training history saved")
        plt.close()
        
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[name.replace('Tomato___', '').replace('_', ' ') for name in self.class_names],
            yticklabels=[name.replace('Tomato___', '').replace('_', ' ') for name in self.class_names],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("📊 Confusion matrix saved")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🌿 PLANT DISEASE DETECTION - TRAINING")
    print("="*60)
    
    trainer = PlantModelTrainer(
        data_path='data/raw/New Plant Diseases Dataset(Augmented)'
    )
    
    # Prepare data
    trainer.prepare_data(batch_size=32)
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train(epochs=20)
    
    # Evaluate
    metrics = trainer.evaluate()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Final Validation Accuracy: {metrics['val_accuracy']*100:.2f}%")
    print(f"⏱️ Total Training Time: {metrics['training_time_seconds']/60:.2f} minutes")


if __name__ == "__main__":
    main()