#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch
import numpy as np

# Fashion MNIST constants
FASHION_MNIST_LABELS = 10
FASHION_MNIST_CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--epochs", default=9, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load Fashion MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Reshape data
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Split test data into validation and test sets
    split = len(x_test) // 2
    x_val, y_val = x_test[:split], y_test[:split]
    x_test, y_test = x_test[split:], y_test[split:]
    
    # --- Create the model using the Functional API ---
    inputs = keras.Input(shape=[28, 28, 1], name="input_image")

    # Initial rescaling layer
    x = keras.layers.Rescaling(1 / 255)(inputs)

    x = keras.layers.Conv2D(filters=100,kernel_size=3,strides=1, padding="same",use_bias=False,name="conv1")(x)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Activation("relu", name="relu1")(x)
    x = keras.layers.Dropout(0.35, name="dropout1")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(FASHION_MNIST_LABELS, activation="softmax", name="output_dense")(x)

    # Define the complete model
    model = keras.Model(inputs=inputs, outputs=outputs, name="fashion_mnist_model")
    # --- End of Functional API model creation ---

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    # Train the model
    print("Starting model training...")
    logs = model.fit(
        x_train, y_train,
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(x_val, y_val),
        verbose=1 # Set to 1 or 2 for progress updates
    )

    # Save the model
    save_path = "fashion_mnist_model2.keras"
    print(f"\nTraining finished. Saving model to {save_path}")
    model.save(save_path)
    print("Model saved successfully.")

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}

# --- Prediction Function (Example) ---
def predict_single_functional():
    model_path = "fashion_mnist_model.keras"
    print(f"\nLoading model from {model_path}...")
    try:
        # Loading a functional model saved this way doesn't require custom objects
        loaded_model = keras.saving.load_model(model_path)
        print("Model loaded successfully.")
        loaded_model.summary() # Verify architecture

        # --- Prepare a Single Image for Prediction ---
        # Load data again for demonstration purposes
        (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Pick a single example
        index = 0
        single_image = x_test[index]
        actual_label = y_test[index]
        
        # Add batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
        image_for_prediction = np.expand_dims(single_image, axis=0)

        # --- Make the Prediction ---
        predictions = loaded_model.predict(image_for_prediction)
        predicted_class = np.argmax(predictions[0])

        print(f"\nPredicted class: {predicted_class} ({FASHION_MNIST_CLASS_NAMES[predicted_class]})")
        print(f"Actual class: {actual_label} ({FASHION_MNIST_CLASS_NAMES[actual_label]})")

    except Exception as e:
        print(f"Error loading or predicting: {e}")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # Train and save the functional model
    main(args)
    # Example: Load and predict using the saved functional model
    predict_single_functional()
