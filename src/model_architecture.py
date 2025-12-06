"""
Model architecture for plant disease detection
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0


def create_model(num_classes=10, input_shape=(224, 224, 3)):
    """
    Create plant disease classification model
    
    Args:
        num_classes: Number of disease classes
        input_shape: Input image shape
        
    Returns:
        Compiled model
    """
    # Load EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='PlantDiseaseDetector')
    
    return model, base_model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model
    
    Args:
        model: Keras model
        learning_rate: Learning rate
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    return model


def get_callbacks(model_path):
    """
    Get training callbacks
    
    Args:
        model_path: Path to save model
        
    Returns:
        List of callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    model, base = create_model(num_classes=10)
    model = compile_model(model)
    
    print("✅ Model created successfully!")
    print(f"📊 Total parameters: {model.count_params():,}")