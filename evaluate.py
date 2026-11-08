from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 1. Image filename
filename = "8_img_.png"

# 2. Image paths
paths = {
    "Original": Path("data/raw-890") / filename,
    "Gray-World": Path("results/gray_world") / filename,
    "CLAHE": Path("results/clahe") / filename,
    "Gamma": Path("results/gamma") / filename,
    "Combined": Path("results/combined") / filename,
}

reference_path = Path("data/reference-890") / filename

# 3. Load reference image
reference = cv2.imread(str(reference_path))

if reference is None:
    raise FileNotFoundError(
        f"Reference image not found: {reference_path}"
    )

# 4. Calculate PSNR and SSIM
results = []
images = {}

for method, path in paths.items():
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if image.shape != reference.shape:
        raise ValueError(
            f"Image size mismatch: {method}"
        )

    psnr = peak_signal_noise_ratio(
        reference, image, data_range=255
    )

    ssim = structural_similarity(
        reference,
        image,
        channel_axis=2,
        data_range=255
    )

    results.append((method, psnr, ssim))
    images[method] = image

# 5. Print evaluation results
print("\nUNDERWATER IMAGE ENHANCEMENT EVALUATION")
print("-" * 52)
print(f"{'Method':<18} {'PSNR (dB)':>12} {'SSIM':>12}")
print("-" * 52)

for method, psnr, ssim in results:
    print(f"{method:<18} {psnr:>12.2f} {ssim:>12.4f}")

print("-" * 52)

# 6. Find best methods
best_psnr = max(results, key=lambda x: x[1])
best_ssim = max(results, key=lambda x: x[2])

print("\nHighest PSNR:", best_psnr[0])
print("Highest SSIM:", best_ssim[0])

# 7. Display all images
display_order = [
    "Original",
    "Gray-World",
    "CLAHE",
    "Gamma",
    "Combined"
]

plt.figure(figsize=(16, 9))

for i, method in enumerate(display_order):
    image_rgb = cv2.cvtColor(
        images[method], cv2.COLOR_BGR2RGB
    )

    plt.subplot(2, 3, i + 1)
    plt.imshow(image_rgb)
    plt.title(method)
    plt.axis("off")

reference_rgb = cv2.cvtColor(
    reference, cv2.COLOR_BGR2RGB
)

plt.subplot(2, 3, 6)
plt.imshow(reference_rgb)
plt.title("UIEB Reference")
plt.axis("off")

plt.tight_layout()

# 8. Save comparison figure
output_dir = Path("results/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

figure_path = output_dir / "comparison.png"
plt.savefig(str(figure_path), dpi=200)

print("\nComparison figure saved:", figure_path)

plt.show()
