import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

INPUT_ROOT = "/kaggle/input/brain-tumor-mri-dataset"
OUTPUT_ROOT = "/content/preprocessed_anisotropic"

def anisotropic_diffusion(img, niter=15, kappa=30, gamma=0.1):
    img = img.astype(np.float32)
    for _ in range(niter):
        nN = np.roll(img, -1, axis=0) - img
        nS = np.roll(img,  1, axis=0) - img
        nE = np.roll(img, -1, axis=1) - img
        nW = np.roll(img,  1, axis=1) - img

        cN = np.exp(-(nN/kappa)**2)
        cS = np.exp(-(nS/kappa)**2)
        cE = np.exp(-(nE/kappa)**2)
        cW = np.exp(-(nW/kappa)**2)

        img += gamma * (cN*nN + cS*nS + cE*nE + cW*nW)

    return np.clip(img, 0, 255).astype(np.uint8)

def process_image(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    out = anisotropic_diffusion(img)
    cv2.imwrite(dst, out)
    return True

pairs = []
for root, _, files in os.walk(INPUT_ROOT):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
            src = os.path.join(root, f)
            rel = os.path.relpath(src, INPUT_ROOT)
            dst = os.path.join(OUTPUT_ROOT, rel)
            pairs.append((src, dst))

print("📂 Images found:", len(pairs))


results = Parallel(n_jobs=-1)(
    delayed(process_image)(s, d) for s, d in tqdm(pairs)
)

print("✅ Preprocessing completed!")