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

# Gamma correction
gamma = 0.6

# Create gamma correction lookup table
table = np.array([
    ((i / 255.0) ** gamma) * 255
    for i in range(256)
]).astype(np.uint8)

# Apply gamma correction
enhanced = cv2.LUT(original, table)

# Save enhanced image
output_dir = Path("results/gamma")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / filename
cv2.imwrite(str(output_path), enhanced)

print("Saved Gamma image:", output_path)
print("Gamma value:", gamma)

# Convert BGR images to RGB for display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

# Display three images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Underwater Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(enhanced_rgb)
plt.title("Gamma Correction (gamma=0.6)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reference_rgb)
plt.title("UIEB Reference Image")
plt.axis("off")

plt.tight_layout()
plt.show()
