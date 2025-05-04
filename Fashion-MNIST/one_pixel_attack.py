import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch") 

import keras
import torch
import numpy as np
import matplotlib.pyplot as plt
FASHION_MNIST_CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def apply_pixels(img, vec, d):
    adv = img.copy()
    H,W,_ = img.shape
    for i in range(d):
        x,y,v = vec[3*i:3*i+3]
        xi, yi = int(np.clip(round(x),0,W-1)), int(np.clip(round(y),0,H-1))
        adv[yi,xi,0] = np.clip(v,0,1)
    return adv

def one_pixel_attack(model, img, true, d=1, pop=50, iters=100):
    img = img.astype(np.float32) / 255.0
    dims = 3 * d
    popv = np.random.rand(pop, dims)
    bounds = np.array([img.shape[1], img.shape[0], 1.0])
    bounds_tile = np.tile(bounds, d)

    for gen in range(iters):
        # Build adversarial batch
        adv_batch = np.stack([apply_pixels(img, popv[i] * bounds_tile, d) for i in range(pop)])
        # Batch predict
        probs = model.predict(adv_batch*255, verbose=0)  # shape (pop, num_classes)
        fitness = probs[:, true]  # minimize P(true)

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        print(f"Gen {gen+1}/{iters} — best fitness = {best_fitness:.4f}")

        adv_best = adv_batch[best_idx]
        pred_best = np.argmax(probs[best_idx])
        if pred_best != true:
            return img, adv_best, true, pred_best

        # DE/rand/1 mutation + binomial crossover
        a, b, c = np.random.choice(pop, 3, replace=False)
        mutant = popv[a] + 0.5 * (popv[b] - popv[c])
        cross = np.random.rand(pop, dims) < 0.9
        popv = np.where(cross, mutant, popv)
        popv = np.clip(popv, 0, 1)

    # If no success
    return img, None, true, None


if __name__=='__main__':
    model = keras.saving.load_model("fashion_mnist_model2.keras")
    (_,_), (X, y) = keras.datasets.fashion_mnist.load_data()
    n = 32
    x0 = np.expand_dims(X[n], -1)
    true_label = y[n]
    pred_before = np.argmax(model.predict(np.expand_dims(x0, 0), verbose=0))
    d = 4
    orig, adv, orig_lbl, adv_lbl = one_pixel_attack(model, x0, pred_before, d, pop=50000, iters=200)

    if adv is not None:
        pred_after = np.argmax(model.predict(np.expand_dims(adv*255, 0), verbose=0))
        print(f"Attack succeeded: True label = {true_label}, Pred before = {pred_before}, Pred after = {pred_after}")
        
        # Show original and adversarial images
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(x0.squeeze(), cmap='gray')
        axs[0].set_title(f'Original (Pred: {FASHION_MNIST_CLASS_NAMES[pred_before]})')
        axs[0].axis('off')

        axs[1].imshow(adv.squeeze(), cmap='gray')
        axs[1].set_title(f'{d} pixels changed (Pred: {FASHION_MNIST_CLASS_NAMES[pred_after]})')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print("Attack failed")