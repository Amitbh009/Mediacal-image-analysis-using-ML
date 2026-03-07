# snr_compute.py

import cv2
import os
import numpy as np
import glob

# -------------------------------
# CONFIG - Set your dataset folder
# -------------------------------
DATASET_FOLDER = "/content/preprocessed_anisotropic"  # root folder
OUTPUT_FILE = "snr_results.csv"

# -------------------------------
# Function to compute SNR for one image
# -------------------------------
def compute_snr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Cannot read {image_path}")
        return None
    # brain pixels: ignore background (assume 0 = background)
    brain_pixels = img[img > 0]
    if len(brain_pixels) == 0:
        return 0
    mean_intensity = np.mean(brain_pixels)
    std_intensity = np.std(brain_pixels)
    snr = mean_intensity / (std_intensity + 1e-8)  # avoid division by zero
    return snr

# -------------------------------
# Loop through all images
# -------------------------------
results = []
for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(root, file)
            snr = compute_snr(path)
            if snr is not None:
                results.append((path, snr))

# -------------------------------
# Save results to CSV
# -------------------------------
import csv
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image_Path", "SNR"])
    writer.writerows(results)

print(f"✅ SNR computation complete! Results saved to {OUTPUT_FILE}")
