# 🧨 One‑Pixel & Few‑Pixel Attack on CIFAR‑10 (Su et al., 2019)

This project implements a **faithful, high-efficiency re-implementation** of the *one‑pixel adversarial attack* proposed by Su, Vargas, and Sakurai (2019), targeting a small convolutional neural network trained on CIFAR‑10. It includes:

* A lightweight CNN training script
* A custom **Differential Evolution (DE)** attacker supporting 1, 3, or more pixel perturbations
* A batch runner that creates a full Markdown + PNG report
* An interactive demo script

---

## 🚀 Quickstart

```bash
cd one_pixel_attack_demo

# 1) Install dependencies
pip install -r requirements.txt

# 2) Train the CNN (~3 minutes on Apple Silicon / GPU)
python train_model.py --epochs 20

# 3) Run a one-pixel attack demo on test image #0
python demo.py --index 0
```

---

## 🧠 What’s Inside

### 🔹 Model: `SimpleCNN`

A compact 2-layer convolutional network with ≈80K parameters for CIFAR‑10 (32×32×3).
Training via `train_model.py` achieves \~70% test accuracy in 20 epochs.

```bash
python train_model.py --epochs 20 --lr 0.001 --batch_size 128
```

Weights are saved to:

```
checkpoints/cifar_simplecnn.pth
```

---

### 🔹 One-Pixel / Few-Pixel Attack

File: [`one_pixel_attack.py`](./one_pixel_attack.py)
A **from-scratch implementation** of Differential Evolution that:

* Matches the hyperparameters in the original paper
* Uses **batch vectorized inference** (40× faster on GPU)
* Supports **1, 3, 5+ pixels**
* Stops early when confidence drops below a threshold
* Does not rely on external optimizers like SciPy

You can call it programmatically via:

```python
from one_pixel_attack import one_pixel_attack
adv_img, adv_lbl, adv_probs, success = one_pixel_attack(
    model, image, true_label,
    pixels=1, device='cuda', max_iter=100, popsize=400
)
```

Returns:

* `adv_img`: perturbed numpy image `[C,H,W]` in `[0,1]`
* `adv_lbl`: predicted class of the adversarial example
* `adv_probs`: softmax vector
* `success`: whether attack succeeded

---

### 🔹 Demo Script

File: [`demo.py`](./demo.py)
Visualises a **side-by-side comparison** of original and adversarial images for a single CIFAR‑10 test image.

```bash
python demo.py --index 0 --pixels 3 --max_iter 200 --popsize 400
```

#### Options:

| Argument     | Description                           |
| ------------ | ------------------------------------- |
| `--index`    | Which test image to attack (0–9999)   |
| `--pixels`   | Number of pixels to modify (1, 3, 5…) |
| `--max_iter` | DE generations (default: 100)         |
| `--popsize`  | Population size (default: 400)        |
| `--device`   | Choose `"cpu"` or `"cuda"`            |
| `--restarts` | Independent DE restarts (default: 1)  |

---

### 🔹 Batch Attack + Markdown Report

File: [`batch_attack.py`](./batch_attack.py)
Runs the attack across the **first 200 test images** for multiple pixel budgets.

Creates:

* `results/results.md`: Markdown summary
* `results/png/*.png`: before/after images

```bash
python batch_attack.py
```

#### Adjustable config:

Modify `PIXEL_BUDGETS`, `MAX_ITER`, `POPSIZE` in the script directly.

Each image row contains:

* Index
* True → predicted label
* Pixel budget used
* CLI command for reproducibility
* Original and adversarial thumbnails

---

## 📈 Example Output

<img src="results/png/idx_0_orig.png" height="64"/> → <img src="results/png/idx_0_adv_px1.png" height="64"/>

| idx | orig → pred | variant (pixels) | cmd | result | image |
|----:|-------------|------------------|-----|--------|-------|
| 0 | dog | 1 | `python demo.py --index 0 --pixels 1 --popsize 400 --max\_iter 500` | ✅ **success** | <img src='png/idx_0_orig.png' height='32'> → <img src="png/idx_0_adv_px1.png" height="32"> |
| 1 | ship | 1 | `python demo.py --index 1 --pixels 1 --popsize 400 --max\_iter 500` | ✅ **success** | <img src='png/idx_1_orig.png' height='32'> → <img src="png/idx_1_adv_px1.png" height="32"> |
| 2 | airplane | 1 | `python demo.py --index 2 --pixels 1 --popsize 400 --max\_iter 500` | ✅ **success** | <img src='png/idx_2_orig.png' height='32'> → <img src="png/idx_2_adv_px1.png" height="32"> |

---

## 🔍 How It Works

* Each candidate solution is a vector:
  **(x, y, R, G, B) × `pixels`**
* Attack goal: **minimize true label probability** (untargeted)
* Optimization via **DE/rand/1/bin** scheme
* Attack succeeds if confidence of true class drops below `early_confidence` or if class flips

### Key Features

* Fully vectorized GPU batch inference
* True-to-paper pixel initialisation: `𝒩(128, 127²)` per channel
* Native early stopping (faster attack)
* Multi-pixel & multi-restart support

---

## 📚 Reference

> Su, Jiawei, Danilo Vargas, and Kouichi Sakurai.
> *"One Pixel Attack for Fooling Deep Neural Networks."*
> IEEE Transactions on Evolutionary Computation, 2019.
> [\[DOI\]](https://doi.org/10.1109/TEVC.2019.2890858)

---

## 🛠 Requirements

* Python 3.9+
* PyTorch
* NumPy
* torchvision
* tqdm
* matplotlib
* Pillow

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
one_pixel_attack_demo/
├── train_model.py          # Train the CNN
├── model.py                # SimpleCNN definition
├── one_pixel_attack.py     # Main attack logic (DE loop)
├── demo.py                 # CLI demo visualiser
├── batch_attack.py         # Markdown + PNG batch report
├── checkpoints/            # Stores trained weights
├── results/                # Markdown + images for report
└── data/                   # Auto-downloaded CIFAR-10
```