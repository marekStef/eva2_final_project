#!/usr/bin/env python3
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
# Assuming your MNIST data loading class is available
from mnist import MNIST # Or however you load MNIST data
from PIL import Image

# --- Load the Saved Model ---
model_path = "model.keras" # Make sure this path is correct
print(f"Loading model from {model_path}...")
# Use keras.saving.load_model to load the model [1][4][5]
loaded_model = keras.saving.load_model(model_path)
print("Model loaded successfully.")

# --- Prepare a Single Image for Prediction ---
# Load MNIST data
mnist = MNIST()
# Get a single image (e.g., the first image from the test set)
# mnist.test.data['images'] likely has shape (num_images, H, W, C)
single_image_path = "adv_img.png" 
single_image = Image.open(single_image_path).convert('L')

# Convert to NumPy array
image_array = np.array(single_image)
single_image = image_array
# The actual label for comparison (optional)
actual_label = mnist.test.data["labels"][0]

# Check the shape of the single image
print(f"Original image shape: {single_image.shape}") # Should be (28, 28, 1) for MNIST

# model.predict expects a batch of images [2]
# We need to add a batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
image_for_prediction = np.expand_dims(single_image, axis=0) # Add batch dimension [2]
print(f"Image shape for prediction: {image_for_prediction.shape}") # Should be (1, 28, 28, 1)

# --- Make the Prediction ---
# Use the predict method of the loaded model [1][4]
predictions = loaded_model.predict(image_for_prediction)
probs = predictions[0] # Get the probabilities for the first image in the batch
# predictions will be an array of probabilities for each class (0-9)
# e.g., [[0.01, 0.02, ..., 0.9, 0.03]] for a single image batch

# --- Interpret the Prediction ---
# Find the index (digit) with the highest probability
pct_strings = [f"{p * 100:.2f}%" for p in probs]
predicted_digit = np.argmax(predictions[0]) # Get the index of the max probability

print(f"\nPredicted digit: {predicted_digit}")
print(f"Actual digit:    {actual_label}")
print("Predictions (percent):", pct_strings)

