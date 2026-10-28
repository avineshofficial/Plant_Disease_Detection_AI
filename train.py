#!/usr/bin/env python3
"""
Model Training Script for Plant Disease Detection AI
Matches the architecture and training pipeline from Model_Training.ipynb.
"""

import os
import sys
import numpy as np

def train_model(dataset_dir="Dataset", epochs=5, batch_size=32, target_size=(225, 225)):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, dataset_dir)
    train_dir = os.path.join(dataset_path, "Train")
    val_dir = os.path.join(dataset_path, "Validation")

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at '{train_dir}'.")
        print("\nTo train the model on the full dataset:")
        print("1. Download the 'Plant Disease Recognition Dataset' from Kaggle:")
        print("   https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset")
        print("2. Extract the archive into a folder named 'Dataset' inside this repository.")
        print("   Expected structure:")
        print("     Dataset/")
        print("       ├── Train/        (Healthy, Powdery, Rust)")
        print("       ├── Validation/   (Healthy, Powdery, Rust)")
        print("       └── Test/         (Healthy, Powdery, Rust)")
        print("3. Run this script again: python train.py")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

    print(f"Loading data from '{train_dir}'...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = None
    if os.path.exists(val_dir):
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

    print(f"Detected class mapping: {train_generator.class_indices}")

    # Build CNN architecture matching Model_Training.ipynb
    model = Sequential([
        Input(shape=(target_size[0], target_size[1], 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print(f"\nStarting model training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    output_path = os.path.join(base_dir, "model.h5")
    model.save(output_path)
    print(f"\nTraining complete. Model saved to '{output_path}'.")


if __name__ == "__main__":
    train_model()

