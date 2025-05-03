# batch_attack.py
# -------------------------------------------------------------------
# Batch runner for the one-pixel / few-pixel attack – generates a
# Markdown report + PNGs for the first 200 CIFAR-10 test images.
# -------------------------------------------------------------------
import os, pathlib, argparse, datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

from model import SimpleCNN
from one_pixel_attack import one_pixel_attack

# --------------------------------------------------------------------- #
#  Config – tweak here if you like
# --------------------------------------------------------------------- #
PIXEL_BUDGETS: List[int] = [1, 3, 5, 7, 10, 12, 14, 16, 18, 20]  # 10 variants
MAX_ITER     = 500
POPSIZE      = 400
RESTARTS     = 1
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR  = Path("results")
PNG_DIR  = OUT_DIR / "png"
MD_PATH  = OUT_DIR / "results.md"
PNG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------- #
def save_tensor_png(t: torch.Tensor, path: Path):
    """C×H×W tensor in [0,1] → PNG file."""
    img = (np.transpose(t.cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def md_escape(s: str) -> str:
    return s.replace("_", "\\_")

# --------------------------------------------------------------------- #
#  Load data & model
# --------------------------------------------------------------------- #
print("Loading CIFAR-10 …")
transform = transforms.Compose([transforms.ToTensor()])
testset   = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

print("Loading model weights …")
model = SimpleCNN()
model.load_state_dict(torch.load("checkpoints/cifar_simplecnn.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# --------------------------------------------------------------------- #
#  Markdown header
# --------------------------------------------------------------------- #
with MD_PATH.open("w", encoding="utf-8") as f_md:
    f_md.write("# One-/Few-Pixel Attack – Batch Results\n\n")
    f_md.write(f"*Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}*\n\n")
    f_md.write("| idx | orig → pred | variant (pixels) | cmd | result | image |\n")
    f_md.write("|----:|-------------|------------------|-----|--------|-------|\n")

# --------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------- #
for idx in range(200):
    image, true_lbl = testset[idx]
    # original prediction
    with torch.no_grad():
        probs = torch.softmax(model(image.unsqueeze(0).to(DEVICE)), dim=1)[0]
    orig_pred = int(torch.argmax(probs))

    # save original png once
    orig_png = PNG_DIR / f"idx_{idx}_orig.png"
    save_tensor_png(image, orig_png)

    success = False
    for px in PIXEL_BUDGETS:
        adv_img, adv_lbl, adv_probs, success = one_pixel_attack(
            model, image, true_lbl,
            device   = DEVICE,
            pixels   = px,
            popsize  = POPSIZE,
            max_iter = MAX_ITER,
            restarts = RESTARTS,
            verbose  = False
        )
        if success:
            adv_png = PNG_DIR / f"idx_{idx}_adv_px{px}.png"
            save_tensor_png(torch.from_numpy(adv_img), adv_png)
            break  # stop escalating pixel budget

    # -----------------------------------------------------------------
    # Markdown row
    # -----------------------------------------------------------------
    cli_cmd = f"`python demo.py --index {idx} --pixels {px} --popsize {POPSIZE} --max_iter {MAX_ITER}`"
    result  = "✅ **success**" if success else "❌ fail"
    with MD_PATH.open("a", encoding="utf-8") as f_md:
        f_md.write(f"| {idx} "
                   f"| {testset.classes[orig_pred]} "
                   f"| {px if success else '—'} "
                   f"| {md_escape(cli_cmd)} "
                   f"| {result} "
                   f"| <img src='png/{orig_png.name}' height='32'>"
                   f" → "
                   f"{'' if not success else f'<img src=\"png/{adv_png.name}\" height=\"32\">'}"
                   f" |\n")

    print(f"[{idx:03}] done – {'success' if success else 'fail'} (first try px={px})")

print(f"\n✅ finished.  Markdown report: {MD_PATH.relative_to(Path.cwd())}")
