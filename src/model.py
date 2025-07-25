# src/model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # Or your chosen base model like MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.regularizers import l2

def build_transfer_model(img_height, img_width, num_classes=1, dropout_rate=0.4, l2_strength=0.001):
    """
    Builds a transfer learning model using ResNet50 as the base.
    Includes a custom classification head with Dropout and L2 regularization.
    The base model's layers are initially frozen.

    Args:
        img_height (int): Input image height.
        img_width (int): Input image width.
        num_classes (int): Number of output classes (1 for binary sigmoid).
        dropout_rate (float): Dropout rate for the dropout layer.
        l2_strength (float): L2 regularization strength.

    Returns:
        tuple: (compiled_model, base_model_instance)
    """
    # Load ResNet50 base model with ImageNet weights, excluding the top (classification) layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Freeze the convolutional base layers
    base_model.trainable = False

    # Define the input layer matching the base model's input
    inputs = Input(shape=(img_height, img_width, 3))

    # Connect the base model to the new input layer.
    # training=False ensures batch_norm/dropout layers in base model run in inference mode for Phase 1
    x = base_model(inputs, training=False)

    # Add custom classification head
    x = GlobalAveragePooling2D()(x) # Flatten features
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_strength))(x) # Dense layer with L2
    x = Dropout(dropout_rate)(x) # Dropout layer
    outputs = Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(l2_strength))(x) # Output layer with L2

    model = Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate):
    """
    Compiles the Keras model with Adam optimizer, BinaryCrossentropy loss, and desired metrics.

    Args:
        model (tf.keras.Model): The Keras model to compile.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    return model

def unfreeze_and_recompile_model(model, base_model, fine_tune_learning_rate, num_unfreeze_layers=20):
    """
    Unfreezes specified layers of the base model and recompiles the entire model
    with a lower learning rate for fine-tuning.

    Args:
        model (tf.keras.Model): The full Keras model.
        base_model (tf.keras.Model): The pre-trained base model part of the full model.
        fine_tune_learning_rate (float): Learning rate for fine-tuning.
        num_unfreeze_layers (int): Number of top layers in the base_model to unfreeze.

    Returns:
        tf.keras.Model: The recompiled Keras model.
    """
    # Unfreeze a portion of the base model's layers
    # Typically, keep BatchNormalization layers frozen during fine-tuning unless dataset is very large
    for layer in base_model.layers[-num_unfreeze_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # Recompile the model with a much lower learning rate for fine-tuning
    # This reinitializes the optimizer state, so learning rate callbacks will reset
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )
    return model