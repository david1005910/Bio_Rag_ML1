#!/usr/bin/env python3
"""Test script for DeepIMAGER with pre-trained dendritic model."""

import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import numpy as np

print("=" * 60)
print("DeepIMAGER Dendritic Model Test")
print("=" * 60)

# Import Keras
try:
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import SGD
    print("Using TensorFlow backend")
except ImportError:
    import keras
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from keras.optimizers import SGD
    print("Using Keras 3 with PyTorch backend")

# Import model components
sys.path.insert(0, '.')
from model.resnet50 import Conv_BN_Relu, resiidual_c_or_d

print(f"\nKeras version: {keras.__version__}")

def build_model(n_neighbors=11, img_size=32):
    """Build the DeepIMAGER model architecture."""

    def get_resnet50_branch(input_shape):
        """Create a ResNet50-style branch."""
        input_img = keras.layers.Input(shape=input_shape)
        conv1 = Conv_BN_Relu(64, (7, 7), 1, input_img)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)

        filters = 64
        num_residuals = [3, 4, 6, 3]
        for i, num_residual in enumerate(num_residuals):
            for j in range(num_residual):
                if j == 0:
                    x = resiidual_c_or_d(x, filters, 'd')
                else:
                    x = resiidual_c_or_d(x, filters, 'c')
            filters = filters * 2

        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        x = Dropout(0.5)(x)
        model_out = Dense(512)(x)

        return keras.Model(input_img, model_out)

    # Single image branch (primary TF)
    single_shape = (img_size, img_size, 1)
    single_model = get_resnet50_branch(single_shape)

    # Multi-image branch (neighbors)
    pair_shape = (img_size, img_size, 1)
    pair_model = get_resnet50_branch(pair_shape)

    # Build combined model
    input_single = keras.layers.Input(shape=single_shape)
    single_out = single_model(input_single)

    input_list = [input_single]
    pair_outputs = []

    for i in range(n_neighbors - 1):
        inp = keras.layers.Input(shape=pair_shape)
        input_list.append(inp)
        pair_outputs.append(pair_model(inp))

    merged = keras.layers.concatenate(pair_outputs, axis=-1)
    combined = keras.layers.concatenate([single_out, merged], axis=-1)
    combined = Dropout(0.5)(combined)
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = keras.Model(input_list, output)
    sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def main():
    # Model parameters (matching dendritic training config)
    n_neighbors = 11  # 1 primary + 10 neighbor images
    img_size = 32
    batch_size = 4

    print(f"\nModel configuration:")
    print(f"  - Image size: {img_size}x{img_size}")
    print(f"  - Number of inputs: {n_neighbors} (1 primary + {n_neighbors-1} neighbors)")
    print(f"  - Test batch size: {batch_size}")

    # Build model
    print("\n[1/4] Building model architecture...")
    model = build_model(n_neighbors=n_neighbors, img_size=img_size)
    print(f"  Model parameters: {model.count_params():,}")

    # Check for pre-trained weights
    weight_path = "model.h5/dendritic.h5"
    print(f"\n[2/4] Loading pre-trained weights from {weight_path}...")

    if os.path.exists(weight_path):
        try:
            model.load_weights(weight_path)
            print("  Weights loaded successfully!")
        except Exception as e:
            print(f"  Warning: Could not load weights: {e}")
            print("  Continuing with random weights for architecture test...")
    else:
        print(f"  Weight file not found, using random weights")

    # Create synthetic test data
    print("\n[3/4] Creating synthetic test data...")
    test_inputs = []
    for i in range(n_neighbors):
        # Generate random 2D histogram-like data
        data = np.random.rand(batch_size, img_size, img_size, 1).astype(np.float32)
        test_inputs.append(data)

    test_labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    print(f"  Created {n_neighbors} input arrays, each shape: {test_inputs[0].shape}")

    # Run prediction
    print("\n[4/4] Running model prediction...")
    predictions = model.predict(test_inputs, verbose=0)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions.flatten()}")

    # Evaluate
    loss, accuracy = model.evaluate(test_inputs, test_labels, verbose=0)
    print(f"\n  Test loss: {loss:.4f}")
    print(f"  Test accuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)
    print("TEST PASSED: Model architecture and inference working!")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
