import os
import cv2
import numpy as np
import pywt
from tqdm import tqdm

INPUT_ROOT = "/content/preprocessed_anisotropic"
OUTPUT_ROOT = "/content/wavelet_enhanced"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def wavelet_enhancement(img, wavelet='db4', level=1, alpha=1.5):
    img = img.astype(np.float32)

    coeffs = pywt.wavedec2(img, wavelet, level=level)
    cA = coeffs[0]
    details = coeffs[1:]

    enhanced_details = []
    for (cH, cV, cD) in details:
        enhanced_details.append((
            cH * alpha,
            cV * alpha,
            cD * alpha
        ))

    enhanced_img = pywt.waverec2([cA] + enhanced_details, wavelet)
    enhanced_img = np.clip(enhanced_img, 0, 255)

    return enhanced_img.astype(np.uint8)

count = 0
for root, _, files in os.walk(INPUT_ROOT):
    for f in tqdm(files):
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
            src = os.path.join(root, f)
            rel = os.path.relpath(src, INPUT_ROOT)
            dst = os.path.join(OUTPUT_ROOT, rel)

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            enhanced = wavelet_enhancement(img)
            cv2.imwrite(dst, enhanced)
            count += 1

print(f"✅ Wavelet enhancement completed for {count} images")
