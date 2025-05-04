why this code seems to not be worknig properly the resulted image is classfied correctly:
#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch") 

import keras
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from mnist import MNIST
# mnist.test.data['images'].shape
# nist.test.data["images"]
# first_label = mnist.test.data["labels"][n]


def predict_keras(model: keras.Model, batch_bhwc: np.ndarray) -> np.ndarray:
    """
    Get softmax probabilities for a batch using a Keras model.
    Assumes input batch is in BHWC format, scaled [0, 1].
    Returns probabilities as a NumPy array.
    """
    # Keras predict handles batching and returns numpy arrays
    probabilities = model.predict(batch_bhwc)
    return probabilities

# --- Pixel Application Helper (Adapted for Keras HWC Grayscale) ---
def apply_pixel_modification_keras(base_hwc: np.ndarray,
                                   de_vector: np.ndarray,
                                   pixels: int) -> np.ndarray:
    """
    Return a perturbed copy of base_hwc according to DE vector de_vector.
    Assumes base_hwc is a NumPy array [Height, Width, 1] with values in [0, 1].
    DE vector format: [x1, y1, value1, x2, y2, value2, ...]. Values are [0, 1].
    """
    adv_img = base_hwc.copy()
    H, W, C = base_hwc.shape
    if C != 1:
        raise ValueError("apply_pixel_modification_keras expects grayscale HWC image (C=1)")

    for p in range(pixels):
        # Extract x, y, and the new pixel value (scaled 0-1)
        px, py, p_value = de_vector[p * 3 : p * 3 + 3]

        # Round coordinates to nearest integer and clip to image bounds
        x_coord = int(np.clip(round(px), 0, W - 1))
        y_coord = int(np.clip(round(py), 0, H - 1))

        # Clip pixel value to [0, 1]
        pixel_value = np.clip(p_value, 0.0, 1.0)

        # Apply the modification to the single channel
        adv_img[y_coord, x_coord, 0] = pixel_value

    return adv_img

# --- One-Pixel Attack Function (Adapted for Keras) ---
def one_pixel_attack_keras(model: keras.Model,
                           image_hwc: np.ndarray, # Expects HWC format, [0, 1]
                           true_label: int,
                           *,
                           pixels: int = 4,       # d in the paper
                           targeted: bool = False,
                           target_class: int | None = None,
                           popsize: int = 100,    # Smaller popsize for faster demo
                           max_iter: int = 700,    # Fewer iterations for faster demo
                           restarts: int = 1,
                           F_scale: float = 0.5,
                           crossover_prob: float = 0.9,
                           early_confidence_threshold: float | None = 0.05, # Stop if P(true_label) drops below this
                           verbose: bool = True
                           ) -> Tuple[np.ndarray | None, int, np.ndarray | None, bool]:
    """
    Perform DE one-pixel attack using a Keras model.
    Return (adv_img HWC float32 [0,1] | None, adv_label, adv_probs | None, success).
    """
    H, W, C = image_hwc.shape
    if C != 1:
        raise ValueError("one_pixel_attack_keras expects grayscale HWC image (C=1)")

    dims = 3 * pixels  # (x, y, value) for each pixel
    bounds_x = (0, W - 1)
    bounds_y = (0, H - 1)
    bounds_value = (0.0, 1.0) # Pixel values are scaled [0, 1]

    rng = np.random.default_rng()

    best_adv_img = None
    best_probs = None
    best_pred_label = -1
    attack_success = False

    if targeted and target_class is None:
        raise ValueError("Target class must be specified for targeted attacks.")
    if targeted and target_class == true_label:
        print("Warning: Target class is the same as the true label.")

    def evaluate_population(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised fitness & probs for the whole population."""
        # 1. Apply pixel modifications to create a batch of adversarial images
        adv_images_batch = np.stack([
            apply_pixel_modification_keras(image_hwc, individual, pixels)
            for individual in population
        ]) # Shape will be (popsize, H, W, C) - correct for Keras

        # 2. Get predictions from the Keras model
        batch_probs = predict_keras(model, adv_images_batch) # Shape (popsize, num_classes)

        # 3. Calculate fitness score
        if targeted:
            # Maximize probability of target class -> Minimize negative probability
            fitness_scores = -batch_probs[:, target_class]
        else:
            # Minimize probability of the true class
            fitness_scores = batch_probs[:, true_label]

        return fitness_scores, batch_probs

    # --- Differential Evolution Loop ---
    for restart in range(restarts):
        if verbose: print(f"\n--- Restart {restart + 1}/{restarts} ---")

        # Initialize Population
        population = np.empty((popsize, dims), dtype=np.float32)
        for i in range(popsize):
            for p in range(pixels):
                idx_base = p * 3
                population[i, idx_base + 0] = rng.uniform(*bounds_x) # x coordinate
                population[i, idx_base + 1] = rng.uniform(*bounds_y) # y coordinate
                population[i, idx_base + 2] = rng.uniform(*bounds_value) # pixel value

        # Evaluate initial population
        fitness, probabilities = evaluate_population(population)
        current_best_fitness = np.min(fitness)
        best_idx_in_restart = np.argmin(fitness)

        for generation in range(max_iter):
            # --- DE/rand/1 Mutation ---
            idxs = np.arange(popsize)
            np.random.shuffle(idxs) # In-place shuffle
            r1, r2, r3 = idxs[:popsize], idxs[popsize:2*popsize] if popsize*2 <= len(idxs) else idxs[:popsize], idxs[2*popsize:3*popsize] if popsize*3 <= len(idxs) else idxs[:popsize] # Get unique indices if possible

            # Ensure r1, r2, r3 are different from the current index i (implicitly handled by random permutation)
            mutant_vectors = population[r1] + F_scale * (population[r2] - population[r3])

            # --- Binomial Crossover ---
            cross_points = rng.random((popsize, dims)) < crossover_prob
            # Ensure at least one parameter is changed (j_rand)
            j_rand = rng.integers(0, dims, size=popsize)
            cross_points[np.arange(popsize), j_rand] = True
            trial_vectors = np.where(cross_points, mutant_vectors, population)

            # --- Apply Bounds (Clipping) ---
            trial_vectors[:, 0::3] = np.clip(trial_vectors[:, 0::3], *bounds_x)   # x
            trial_vectors[:, 1::3] = np.clip(trial_vectors[:, 1::3], *bounds_y)   # y
            trial_vectors[:, 2::3] = np.clip(trial_vectors[:, 2::3], *bounds_value) # value

            # --- Selection ---
            trial_fitness, trial_probabilities = evaluate_population(trial_vectors)
            improved_mask = trial_fitness < fitness
            population[improved_mask] = trial_vectors[improved_mask]
            fitness[improved_mask] = trial_fitness[improved_mask]
            probabilities[improved_mask] = trial_probabilities[improved_mask]

            # Update best solution found so far in this restart
            best_idx_in_gen = np.argmin(fitness)
            if fitness[best_idx_in_gen] < current_best_fitness:
                current_best_fitness = fitness[best_idx_in_gen]
                best_idx_in_restart = best_idx_in_gen
                if verbose:
                    pred_label = np.argmax(probabilities[best_idx_in_restart])
                    print(f"  Gen {generation + 1:3}/{max_iter}: New best fitness={current_best_fitness:.4f} "
                          f"(P(true={true_label})={probabilities[best_idx_in_restart, true_label]:.3f}), "
                          f"Pred={pred_label}")

            # --- Early Stopping Check ---
            best_probs_this_gen = probabilities[best_idx_in_restart]
            pred_label_this_gen = int(np.argmax(best_probs_this_gen))

            stop_condition_met = False
            if targeted:
                # If targeted attack succeeds (prediction is target)
                if pred_label_this_gen == target_class:
                    stop_condition_met = True
            else:
                # If untargeted attack succeeds (prediction is not true label)
                if pred_label_this_gen != true_label:
                    # Check confidence threshold if provided
                    if early_confidence_threshold is None or best_probs_this_gen[true_label] < early_confidence_threshold:
                        stop_condition_met = True

            if stop_condition_met:
                 if verbose: print(f"  Early stopping condition met at gen {generation + 1}.")
                 break # Stop this restart's generation loop

        # --- End of Generations for this restart ---
        # Update overall best if this restart found a better solution OR the first valid one
        best_solution_this_restart = population[best_idx_in_restart]
        best_probs_this_restart = probabilities[best_idx_in_restart]
        best_pred_this_restart = int(np.argmax(best_probs_this_restart))

        # Check if this restart produced a successful attack
        success_this_restart = (
            (not targeted and best_pred_this_restart != true_label) or
            (targeted and best_pred_this_restart == target_class)
        )

        update_overall_best = False
        if success_this_restart:
            if not attack_success: # First successful attack found
                update_overall_best = True
                attack_success = True # Mark that we found at least one success
            elif fitness[best_idx_in_restart] < np.min(evaluate_population(np.array([best_adv_img]))[0]): # Compare fitness
                 # If already successful, check if this one is *better*
                 # Need to re-evaluate fitness of current best_adv_img to compare fairly
                 current_best_fitness_eval = evaluate_population(np.array([best_solution_this_restart]))[0][0]
                 if current_best_fitness_eval < fitness[best_idx_in_restart]:
                      update_overall_best = True

        elif not attack_success: # If no success yet, update with the best-effort result so far
             if best_adv_img is None or fitness[best_idx_in_restart] < evaluate_population(np.array([best_solution_this_restart]))[0][0]:
                  update_overall_best = True


        if update_overall_best:
            if verbose: print(f"  Updating overall best result from restart {restart + 1}.")
            best_adv_img = apply_pixel_modification_keras(image_hwc, best_solution_this_restart, pixels)
            best_probs = best_probs_this_restart
            best_pred_label = best_pred_this_restart

        if attack_success and restarts > 1: # Found a success, no need for more restarts if optimizing for speed
             if verbose: print("  Successful attack found, stopping restarts.")
             break

    # --- End of Restarts ---
    if best_adv_img is None: # Should only happen if max_iter=0 or other edge case
         print("Warning: No adversarial image generated.")
         # Return original image info as fallback
         original_probs = predict_keras(model, np.expand_dims(image_hwc, axis=0))[0]
         return image_hwc, int(np.argmax(original_probs)), original_probs, False


    return best_adv_img, best_pred_label, best_probs, attack_success


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keras MNIST One-Pixel Attack")
    # Add relevant arguments if needed, or use defaults
    parser.add_argument("--model_path", type=str, default="model.keras", help="Path to saved Keras model.")
    parser.add_argument("--pixels", type=int, default=5, help="Number of pixels to attack (d).")
    parser.add_argument("--popsize", type=int, default=10000, help="Population size for DE.")
    parser.add_argument("--max_iter", type=int, default=25, help="Max generations per restart for DE.")
    parser.add_argument("--restarts", type=int, default=1, help="Number of DE restarts.")
    parser.add_argument("--targeted", action="store_true", help="Perform targeted attack.")
    parser.add_argument("--target_class", type=int, default=None, help="Target class for targeted attack.")

    cli_args = parser.parse_args() # Use CLI args

    # --- 1. Load Keras Model ---
    model_path = cli_args.model_path
    print(f"Loading Keras model from: {model_path}")
    try:
        # No custom objects needed if saved from Functional API or properly registered class
        loaded_model = keras.saving.load_model(model_path)
        loaded_model.summary()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model was saved correctly (Functional API or registered custom class).")
        exit()

    # --- 2. Load MNIST Data & Select Image ---
    print("Loading MNIST data...")
    mnist = MNIST()
    # Get the first image from the test set
    # MNIST data is often (N, H, W), needs channel dim. Keras expects HWC.
    # Assuming mnist.load_data() gives something like (N, H, W) uint8
    n = 2
    if len(mnist.test.data['images'].shape) == 3: # (N, H, W)
         first_image_original = mnist.test.data["images"][n] # (H, W)
         first_image_hwc = np.expand_dims(first_image_original, axis=-1) # Add channel -> (H, W, 1)
    elif len(mnist.test.data['images'].shape) == 4: # Already (N, H, W, C)
         first_image_hwc = mnist.test.data["images"][n] # (H, W, C)
    else:
         raise ValueError("Unexpected MNIST image data shape")

    first_label = mnist.test.data["labels"][n]

    image_to_attack = first_image_hwc.astype(np.float32)

    # --- 4. Verify Original Prediction ---
    original_prediction_probs = predict_keras(loaded_model, np.expand_dims(image_to_attack, axis=0))
    original_predicted_label = np.argmax(original_prediction_probs)
    print(f"\nOriginal Image: True Label = {first_label}, Predicted Label = {original_predicted_label}")
    print(f"Original Probabilities: {original_prediction_probs}")

    if original_predicted_label != first_label:
        print("Warning: Model already misclassifies the original image. Attack may not be meaningful.")

    # --- 5. Run the Attack ---
    print(f"\nStarting {cli_args.pixels}-pixel attack...")
    adv_img, adv_pred_label, adv_probs, success = one_pixel_attack_keras(
        model=loaded_model,
        image_hwc=image_to_attack,
        true_label=first_label,
        pixels=cli_args.pixels,
        targeted=cli_args.targeted,
        target_class=cli_args.target_class,
        popsize=cli_args.popsize,
        max_iter=cli_args.max_iter,
        restarts=cli_args.restarts,
        verbose=True
    )

    # --- 6. Display Results ---
    print("\n--- Attack Results ---")
    if adv_img is not None and adv_probs is not None:
        print(f"Attack Success: {success}")
        print(f"Original Label: {first_label}")
        print(f"Adversarial Predicted Label: {adv_pred_label}")
        print(f"Adversarial Probabilities: {adv_probs}")

        # Find the modified pixels (approximate)
        diff = np.abs(adv_img - image_to_attack) * 255.0
        modified_coords = np.where(diff > 1) # Find pixels with significant change
        print(f"Approx. Modified Pixel Coordinates (y, x, channel=0): {list(zip(modified_coords[0], modified_coords[1]))}")


        # Visualize
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_to_attack.squeeze(), cmap='gray', vmin=0, vmax=1) # Use squeeze to remove channel dim for imshow
        plt.title(f"Original (True: {first_label}, Pred: {original_predicted_label})")
        plt.axis('off')
        plt.imsave("original.png",image_to_attack.squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(adv_img.squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.imsave("adv_img.png",adv_img.squeeze(), cmap="gray", vmin=0, vmax=1)
        # Highlight modified pixels roughly
        if len(modified_coords[0]) > 0:
            plt.scatter(modified_coords[1], modified_coords[0], c='red', s=40, marker='x') # x vs y reversed for plt.scatter
        plt.title(f"Adversarial (Pred: {adv_pred_label}, Success: {success})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Attack failed to generate an adversarial image.")
