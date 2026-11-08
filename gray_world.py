from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image paths
filename = "8_img_.png"
raw_path = Path("data/raw-890") / filename
ref_path = Path("data/reference-890") / filename

# Load images
original = cv2.imread(str(raw_path))
reference = cv2.imread(str(ref_path))

if original is None or reference is None:
    raise FileNotFoundError("Image files could not be loaded.")

# Gray-World White Balance
image = original.astype(np.float32)

channel_means = image.mean(axis=(0, 1))
target_mean = channel_means.mean()

scale = target_mean / np.maximum(channel_means, 1e-6)
balanced = image * scale

balanced = np.clip(balanced, 0, 255).astype(np.uint8)

# Convert BGR to RGB for display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
balanced_rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

# Display comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Underwater Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(balanced_rgb)
plt.title("Gray-World White Balance")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reference_rgb)
plt.title("UIEB Reference Image")
plt.axis("off")

plt.tight_layout()
# Save Gray-World enhanced image
output_dir = Path("results/gray_world")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / filename
cv2.imwrite(str(output_path), balanced)

print("Saved enhanced image:", output_path)
plt.show()
