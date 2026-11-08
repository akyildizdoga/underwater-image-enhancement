from pathlib import Path
import cv2
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

# CLAHE contrast enhancement
# Convert the image from BGR to LAB
lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

# Separate the LAB channels
l, a, b = cv2.split(lab)

# Apply CLAHE only to the lightness channel
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
enhanced_l = clahe.apply(l)

# Merge the channels and convert back to BGR
enhanced_lab = cv2.merge((enhanced_l, a, b))
enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Save the enhanced image
output_dir = Path("results/clahe")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / filename
cv2.imwrite(str(output_path), enhanced)

print("Saved CLAHE image:", output_path)

# Convert images to RGB for display
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
plt.title("CLAHE Enhancement")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reference_rgb)
plt.title("UIEB Reference Image")
plt.axis("off")

plt.tight_layout()
plt.show()
