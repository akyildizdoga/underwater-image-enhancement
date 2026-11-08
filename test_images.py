from pathlib import Path
import cv2
import matplotlib.pyplot as plt

raw_dir = Path("data/raw-890")
ref_dir = Path("data/reference-890")

filename = "8_img_.png"

raw_path = raw_dir / filename
ref_path = ref_dir / filename

raw = cv2.imread(str(raw_path))
reference = cv2.imread(str(ref_path))

if raw is None or reference is None:
    raise FileNotFoundError("Image files could not be loaded.")

print("Original image:", raw.shape)
print("Reference image:", reference.shape)

raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
ref_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(raw_rgb)
plt.title("Original Underwater Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ref_rgb)
plt.title("UIEB Reference Image")
plt.axis("off")

plt.tight_layout()
plt.show()
