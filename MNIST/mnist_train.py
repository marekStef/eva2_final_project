#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch
import numpy as np # Added for potential prediction later

# Assuming mnist.py defines the MNIST class with H, W, C, LABELS constants
# and data loading capabilities.
from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
# --cnn argument is now unused since the model is hardcoded
# parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=9, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

# --- Custom Model Class Removed ---

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # --- Create the model using the Functional API ---
    inputs = keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C], name="input_image")

    # Initial rescaling layer
    x = keras.layers.Rescaling(1 / 255)(inputs)

    # --- Hardcoded Model Architecture Layers ---
    filters = 64
    kernel_size = 3
    stride = 1
    padding = "same"

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        use_bias=False,
        name="conv1"
    )(x)

    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Activation("relu", name="relu1")(x)
    x = keras.layers.Dropout(0.35, name="dropout1")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(MNIST.LABELS, activation="softmax", name="output_dense")(x)

    # Define the complete model
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_functional_model")
    # --- End of Functional API model creation ---

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Train the model (same as before)
    print("Starting model training...")
    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        verbose=1 # Set to 1 or 2 for progress updates
    )

    # Save the model (same as before, should work without issues now)
    save_path = "my_final_model_functional.keras" # Use a new name to avoid confusion
    print(f"\nTraining finished. Saving model to {save_path}")
    model.save(save_path)
    print("Model saved successfully.")

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}

# --- Prediction Function (Example) ---
def predict_single_functional():
    model_path = "model.keras" # Load the functional model
    print(f"\nLoading model from {model_path}...")
    try:
        # Loading a functional model saved this way doesn't require custom objects
        loaded_model = keras.saving.load_model(model_path)
        print("Model loaded successfully.")
        loaded_model.summary() # Verify architecture

        # --- Prepare a Single Image for Prediction ---
        mnist = MNIST()
        single_image = mnist.test.data["images"][0]
        actual_label = mnist.test.data["labels"][0]
        # Add batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
        image_for_prediction = np.expand_dims(single_image, axis=0)

        # --- Make the Prediction ---
        predictions = loaded_model.predict(image_for_prediction)
        predicted_digit = np.argmax(predictions[0])

        print(f"\nPredicted digit: {predicted_digit}")
        print(f"Actual digit: {actual_label}")

    except Exception as e:
        print(f"Error loading or predicting: {e}")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # Train and save the functional model
    main(args)
    # Example: Load and predict using the saved functional model
    predict_single_functional()
