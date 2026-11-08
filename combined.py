from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Image paths
filename = "8_img_.png"

raw_path = Path("data/raw-890") / filename
ref_path = Path("data/reference-890") / filename

# 2. Load images
original = cv2.imread(str(raw_path))
reference = cv2.imread(str(ref_path))

if original is None or reference is None:
    raise FileNotFoundError("Image files could not be loaded.")

# 3. Gray-World White Balance
image = original.astype(np.float32)

channel_means = image.mean(axis=(0, 1))
gray_mean = channel_means.mean()

scales = gray_mean / np.maximum(channel_means, 1e-6)

balanced = image * scales
balanced = np.clip(balanced, 0, 255).astype(np.uint8)

# 4. CLAHE on LAB lightness channel
lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)

l_channel, a_channel, b_channel = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_l = clahe.apply(l_channel)

enhanced_lab = cv2.merge(
    (enhanced_l, a_channel, b_channel)
)

combined = cv2.cvtColor(
    enhanced_lab, cv2.COLOR_LAB2BGR
)

# 5. Save enhanced image
output_dir = Path("results/combined")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / filename

if not cv2.imwrite(str(output_path), combined):
    raise RuntimeError("Enhanced image could not be saved.")

print("Saved combined image:", output_path)

# 6. Convert images for display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

# 7. Display comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Underwater Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(combined_rgb)
plt.title("White Balance + CLAHE")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reference_rgb)
plt.title("UIEB Reference Image")
plt.axis("off")

plt.tight_layout()
plt.show()
