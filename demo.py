"""End‑to‑end demonstration of the one‑pixel attack on a CIFAR‑10 image."""
import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import SimpleCNN
from one_pixel_attack import one_pixel_attack

CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    parser = argparse.ArgumentParser(description='One‑pixel attack demo')
    parser.add_argument('--weights', default='checkpoints/cifar_simplecnn.pth', help='Path to trained model weights')
    parser.add_argument('--index', type=int, default=0, help='Index of test image to attack (0‑9999)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_iter', type=int, default=100, help='DE generations')
    parser.add_argument('--popsize', type=int, default=20, help='DE popsize (SciPy definition)')
    args = parser.parse_args()

    # Load test set image
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    image, label = testset[args.index]

    # Load model
    model = SimpleCNN()
    if os.path.isfile(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=args.device))
    else:
        raise FileNotFoundError(f"Weights not found at {args.weights}. Run train_model.py first.")

    # Original prediction
    model.eval().to(args.device)
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(args.device))
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    orig_pred = int(np.argmax(probs))

    print(f'Original prediction: {orig_pred} ({CIFAR10_LABELS[orig_pred]}), true label: {label} ({CIFAR10_LABELS[label]})')

    # Run attack
    adv_img, adv_label, adv_probs, success, _ = one_pixel_attack(
        model, image, label, device=args.device, max_iter=args.max_iter, popsize=args.popsize
    )
    print(f'Adversarial prediction: {adv_label} ({CIFAR10_LABELS[adv_label]}), success={success}')

    # Visualise
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.transpose(image.numpy(), (1, 2, 0)))
    axs[0].set_title(f'Original\npred: {CIFAR10_LABELS[orig_pred]}')
    axs[0].axis('off')

    axs[1].imshow(np.transpose(adv_img, (1, 2, 0)))
    axs[1].set_title(f'Adversarial\npred: {CIFAR10_LABELS[adv_label]}')
    axs[1].axis('off')
    plt.suptitle('One‑pixel attack')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
