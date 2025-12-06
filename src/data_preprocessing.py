"""
Data preprocessing for plant disease detection
"""
import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import json


class PlantDataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path, img_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            raw_data_path: Path to raw dataset
            processed_data_path: Path for processed data
            img_size: Target image size
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.img_size = img_size
        
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Define the 10 classes we'll use (all tomato diseases)
        self.selected_classes = [
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
    
    def create_data_generators(self, train_path, val_path, batch_size=32):
        """
        Create data generators for training
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            batch_size: Batch size
            
        Returns:
            train_gen, val_gen
        """
        from keras.preprocessing.image import ImageDataGenerator
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.selected_classes,
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.selected_classes,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def get_class_info(self):
        """Get class names and count"""
        return {
            'num_classes': len(self.selected_classes),
            'class_names': self.selected_classes,
            'class_mapping': {i: name for i, name in enumerate(self.selected_classes)}
        }


if __name__ == "__main__":
    preprocessor = PlantDataPreprocessor(
        raw_data_path='data/raw/New Plant Diseases Dataset(Augmented)',
        processed_data_path='data/processed'
    )
    
    print("✅ Preprocessor initialized!")
    print(f"📊 Classes: {preprocessor.get_class_info()['num_classes']}")