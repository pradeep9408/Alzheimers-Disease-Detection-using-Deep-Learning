import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 224  # ConvNeXt models expect 224x224 images
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 2e-4
FINE_TUNE_LEARNING_RATE = 5e-5
CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_DIR = 'models'
DATASET_DIR = 'data/OriginalDataset'  # Matches your existing dataset folder

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def create_convnext_base_model():
    """Create a model using ConvNeXtBase architecture"""
    # Load the ConvNeXtBase model with pre-trained ImageNet weights
    base_model = ConvNeXtBase(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.LayerNormalization(),
        layers.Dense(1280, activation='gelu'),
        layers.Dropout(0.25),
        layers.LayerNormalization(),
        layers.Dense(640, activation='gelu'),
        layers.Dropout(0.15),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model with Adam optimizer
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Removed ConvNeXtSmall model creation; only ConvNeXtBase is used

def prepare_data():
    """Prepare and augment the dataset"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = valid_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_model():
    """Train the model using ConvNeXtBase architecture"""
    # Always use ConvNeXtBase
    model, base_model = create_convnext_base_model()
    model_name = 'convnext_base_alzheimers'
    
    # Prepare data
    train_generator, validation_generator = prepare_data()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model with frozen base layers
    print(f"Training {model_name} with frozen base layers...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )
    
    # Fine-tuning: unfreeze some layers of the base model
    print(f"Fine-tuning {model_name}...")
    # Unfreeze the last 50 layers of ConvNeXtBase
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with unfrozen layers
    fine_tune_history = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, f'{model_name}_final.h5'))
    
    # Combine histories
    combined_history = {}
    for key in history.history:
        combined_history[key] = history.history[key] + fine_tune_history.history[key]
    
    # Plot training history
    plot_training_history(combined_history, model_name)
    
    return model, combined_history

def plot_training_history(history, model_name):
    """Plot training & validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history['accuracy'])
    ax1.plot(history['val_accuracy'])
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Loss
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title(f'{model_name} Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'{model_name}_history.png'))
    plt.close()

def evaluate_model(model, validation_generator):
    """Evaluate the model on the validation set"""
    results = model.evaluate(validation_generator)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    return results

def main():
    """Main function to train ConvNeXtBase model only"""
    print("Starting Alzheimer's detection model training...")
    
    # Train ConvNeXtBase model only
    print("\n=== Training ConvNeXtBase Model ===")
    model, history = train_model()
    
    print("\nTraining completed successfully!")
    print("Model saved in the 'models' directory.")

if __name__ == "__main__":
    main()
