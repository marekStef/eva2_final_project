# One‑Pixel Attack Demo (Su et al. 2019)

This mini‑project gives you a **fully functional demonstration** of the one‑pixel attack described in  
*Su, Vargas & Sakurai — “One Pixel Attack for Fooling Deep Neural Networks” (2019)*.

## ️📦  Quick start

```bash
git clone <this‑repo> one_pixel_demo   # or download & unzip
cd one_pixel_demo

# 1) install deps (Python 3.9+, macOS / Linux):
pip install -r requirements.txt

# 2) train a small CIFAR‑10 CNN (~3 min on Apple M‑series):
python train_model.py --epochs 20

# 3) run the attack demo on test image #0:
python demo.py --index 0
```

The script will print the model’s original prediction, run Differential Evolution to find the adversarial 1‑pixel
change, then pop up a Matplotlib window showing **before** and **after** images side‑by‑side.

![screenshot](screenshot.png)

### Command‑line options
* `--index N` – choose a different test image (0‒9999).  
* `--max_iter K` – Differential Evolution generations (default 100).  
* `--popsize P` – DE popsize (SciPy definition, default 20).  
* `--device {cpu,cuda}` – force CPU/GPU.  

## How it works (high‑level)
* **Encoding**: one candidate = `(x, y, R, G, B)` describing a single pixel change.  
* **Search**: SciPy’s `differential_evolution` explores that 5‑D space, minimising the true‑class probability.  
* The classifier in `model.py` is a tiny 2‑conv CNN (~80 k params) – good enough to reach ~70 % accuracy after 20 epochs.  
* Attack typically succeeds ~60‑70 % of the time within 7 500 evaluations (100 generations × 75 indiv).  

## 📚 References
* Su, Jiawei, Danilo Vargas, and Kouichi Sakurai. “One Pixel Attack for Fooling Deep Neural Networks.” IEEE Transactions on Evolutionary Computation (2019).
