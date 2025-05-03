"""One‑pixel attack implementation using Differential Evolution (Su et al. 2019)."""
import numpy as np
import torch
from scipy.optimize import differential_evolution


def _predict(model, img, device='cpu'):
    """Return softmax probabilities for a single image (numpy HWC or CHW in [0,1])."""
    if img.ndim == 3:  # CHW
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    else:
        raise ValueError('img must be CHW numpy array')
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs


def one_pixel_attack(model, image, label, device='cpu', max_iter=300, popsize=30,
                     targeted=False, target_class=None, verbose=False):
    """Perform the one‑pixel (ℓ₀=1) attack.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier.
    image : torch.Tensor
        Clean image tensor (C×H×W) in [0,1].
    label : int
        True label index.
    device : str
        'cpu' or 'cuda'.
    max_iter : int
        DE generations.
    popsize : int
        DifferentialEvolution popsize (SciPy definition = individuals / dim).
    targeted : bool
        Whether to do targeted attack.
    target_class : int or None
        Desired target class for targeted attack.
    verbose : bool
        If True, print progress.
    Returns
    -------
    adv_img : numpy.ndarray
        Adversarial image CHW in [0,1].
    adv_label : int
        Model prediction on adv_img.
    probs : numpy.ndarray
        Softmax probs of adv_img.
    success : bool
        Whether attack succeeded.
    result : OptimizeResult
        Raw scipy result object.
    """
    model.eval().to(device)
    img_np = image.cpu().numpy()
    H, W = img_np.shape[1:]
    bounds = [(0, W - 1), (0, H - 1), (0, 255), (0, 255), (0, 255)]

    true_label = label

    def de_obj(x):
        xi, yi, r, g, b = x
        xi = int(round(xi))
        yi = int(round(yi))
        perturbed = img_np.copy()
        perturbed[:, yi, xi] = np.array([r, g, b]) / 255.0
        probs = _predict(model, perturbed, device)

        if targeted and target_class is not None:
            return -probs[target_class]  # maximise target class prob
        else:
            return probs[true_label]  # minimise true class prob

    result = differential_evolution(
        de_obj,
        bounds,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-5,
        polish=False,
        recombination=0.7,
        mutation=(0.5, 1),
        disp=verbose,
    )

    xi, yi, r, g, b = result.x
    xi, yi = int(round(xi)), int(round(yi))
    adv = img_np.copy()
    adv[:, yi, xi] = np.array([r, g, b]) / 255.0
    probs = _predict(model, adv, device)
    adv_label = int(np.argmax(probs))
    success = (adv_label != true_label) if not targeted else (adv_label == target_class)
    return adv, adv_label, probs, success, result
