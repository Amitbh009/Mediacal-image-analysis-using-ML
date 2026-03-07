import os
import cv2
import matplotlib.pyplot as plt

def safe_imread(path, grayscale=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)

    if img is None:
        raise ValueError(f"❌ Failed to load image: {path}")

    return img

# -------------------------------
# UPDATE PATHS CORRECTLY
# -------------------------------
img_path = "/kaggle/input/brain-tumor-mri-dataset/Training/glioma/Tr-glTr_0001.jpg"
mask_path = "/content/segmented_output/Training/glioma/masks/Tr-glTr_0001.jpg"
overlay_path = "/content/segmented_output/Training/glioma/overlays/Tr-glTr_0001.jpg"

# -------------------------------
# LOAD IMAGES SAFELY
# -------------------------------
img = safe_imread(img_path)
mask = safe_imread(mask_path, grayscale=True)
overlay = safe_imread(overlay_path)

# Convert BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -------------------------------
# DISPLAY
# -------------------------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original MRI")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()
