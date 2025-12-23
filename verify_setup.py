#!/usr/bin/env python3
"""Verification script to test DeepIMAGER setup."""

import sys

def test_imports():
    """Test all required imports."""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)

    errors = []

    # Core packages
    packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scipy", None),
        ("sklearn", None),
        ("matplotlib", None),
        ("h5py", None),
    ]

    for pkg, alias in packages:
        try:
            exec(f"import {pkg}")
            print(f"  [OK] {pkg}")
        except ImportError as e:
            print(f"  [FAIL] {pkg}: {e}")
            errors.append(pkg)

    # Keras (with PyTorch backend)
    try:
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        print(f"  [OK] keras (version {keras.__version__}, backend: torch)")
    except ImportError as e:
        print(f"  [FAIL] keras: {e}")
        errors.append("keras")

    # PyTorch
    try:
        import torch
        print(f"  [OK] torch (version {torch.__version__})")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        errors.append("torch")

    return errors

def test_model_imports():
    """Test importing project model modules."""
    print("\n" + "=" * 50)
    print("Testing project model imports...")
    print("=" * 50)

    errors = []

    try:
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        sys.path.insert(0, ".")
        from model import resnet_18
        print("  [OK] model.resnet_18")
    except Exception as e:
        print(f"  [FAIL] model.resnet_18: {e}")
        errors.append("resnet_18")

    try:
        from model import resnet50
        print("  [OK] model.resnet50")
    except Exception as e:
        print(f"  [FAIL] model.resnet50: {e}")
        errors.append("resnet50")

    return errors

def test_data_files():
    """Check if sample data files exist."""
    print("\n" + "=" * 50)
    print("Checking data files...")
    print("=" * 50)

    import os

    # Pre-trained models
    models = [
        "model.h5/dendritic.h5",
        "model.h5/mHSC_L.h5",
    ]

    for model in models:
        if os.path.exists(model):
            size = os.path.getsize(model) / (1024 * 1024)
            print(f"  [OK] {model} ({size:.1f} MB)")
        else:
            print(f"  [MISSING] {model}")

    # Data directories
    data_dirs = [
        "data_evaluation/dendritic",
        "data_evaluation/bonemarrow",
        "data_evaluation/single_cell_type",
    ]

    for d in data_dirs:
        if os.path.isdir(d):
            print(f"  [OK] {d}/")
        else:
            print(f"  [MISSING] {d}/")

def test_basic_operations():
    """Test basic numpy/keras operations."""
    print("\n" + "=" * 50)
    print("Testing basic operations...")
    print("=" * 50)

    try:
        import numpy as np
        arr = np.random.rand(32, 64, 64, 1)
        print(f"  [OK] NumPy array creation: shape {arr.shape}")
    except Exception as e:
        print(f"  [FAIL] NumPy operations: {e}")
        return False

    try:
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        from keras import layers, Model, Input

        # Simple model test
        inputs = Input(shape=(64, 64, 1))
        x = layers.Conv2D(32, 3, padding='same')(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)

        print(f"  [OK] Keras model creation: {model.count_params()} params")

        # Test forward pass
        pred = model.predict(arr[:1], verbose=0)
        print(f"  [OK] Model forward pass: output shape {pred.shape}")
    except Exception as e:
        print(f"  [FAIL] Keras operations: {e}")
        return False

    return True

def main():
    print("\nDeepIMAGER Setup Verification")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print()

    import_errors = test_imports()
    model_errors = test_model_imports()
    test_data_files()
    ops_ok = test_basic_operations()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if import_errors:
        print(f"  Import errors: {import_errors}")
    if model_errors:
        print(f"  Model import errors: {model_errors}")

    if not import_errors and not model_errors and ops_ok:
        print("  All tests passed!")
        return 0
    else:
        print("  Some tests failed. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
