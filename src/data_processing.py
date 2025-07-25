# src/data_processing.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import class_weight

def get_image_data_generators(train_dir, val_dir, test_dir, img_height, img_width, batch_size, preprocess_input_fn):
    """
    Sets up and returns ImageDataGenerators for train, validation, and test sets.
    Applies augmentation to training data and only preprocessing to validation/test data.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        test_dir (str): Path to the test data directory.
        img_height (int): Target height for images.
        img_width (int): Target width for images.
        batch_size (int): Batch size for generators.
        preprocess_input_fn (function): Preprocessing function specific to the model (e.g., ResNet50's preprocess_input).

    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    # Data Augmentation for Training Data:
    train_datagen = ImageDataGenerator(
        rotation_range=15,       # Rotate images by a random angle (max 15 degrees)
        width_shift_range=0.1,   # Shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # Shift images vertically (fraction of total height)
        shear_range=0.1,         # Apply shear transformation
        zoom_range=0.1,          # Apply zoom transformation
        horizontal_flip=True,    # Randomly flip images horizontally
        fill_mode='nearest',     # Strategy for filling in new pixels created by transformations
        preprocessing_function=preprocess_input_fn # Apply model-specific preprocessing
        # Consider adding: brightness_range=[0.8, 1.2] for more robustness if needed
    )

    # No Augmentation for Validation and Test Data (only scaling/preprocessing)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_fn # Apply model-specific preprocessing
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary', # 'binary' for 2 classes, 'categorical' for >2 classes
        color_mode='rgb' # Ensure images are treated as 3-channel (grayscale will be converted)
    )

    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False # Do not shuffle test data to maintain order for evaluation
    )

    return train_generator, validation_generator, test_generator

def compute_class_weights(generator):
    """
    Computes class weights for handling class imbalance in a Keras ImageDataGenerator.

    Args:
        generator (ImageDataGenerator): The training data generator.

    Returns:
        dict: A dictionary mapping class indices to their computed weights.
    """
    y_labels = generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_labels),
        y=y_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict

# Example of a utility function (optional, but good for Notebook 3)
def get_class_names(generator):
    """
    Returns class names from a generator, sorted by their integer index.
    """
    class_names = [name for name, index in sorted(generator.class_indices.items(), key=lambda item: item[1])]
    return class_names