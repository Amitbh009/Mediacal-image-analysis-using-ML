import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

INPUT_ROOT = "/content/wavelet_enhanced"
OUTPUT_ROOT = "/content/kmeans_segmented"
VISUALIZATION_ROOT = "/content/kmeans_visualizations"

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(VISUALIZATION_ROOT, exist_ok=True)

def kmeans_segmentation(img, K=3, post_process=True):
    """
    Performs K-means clustering on grayscale image with post-processing
    Returns binary mask of the brightest cluster (assumed to be tumor)
    
    Parameters:
    - img: Input grayscale image
    - K: Number of clusters (3 for brain: CSF, GM/WM, tumor)
    - post_process: Apply morphological operations to clean mask
    """
    
    pixel_vals = img.reshape((-1, 1))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    
    _, labels, centers = cv2.kmeans(
        pixel_vals,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS 
    )
    
    labels = labels.flatten()
    centers = centers.flatten()
    
    tumor_cluster = np.argmax(centers)
    
    segmented = np.zeros_like(labels, dtype=np.uint8)
    segmented[labels == tumor_cluster] = 255
    segmented_img = segmented.reshape(img.shape)
    
    if post_process:
        kernel = np.ones((3, 3), np.uint8)
        segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_OPEN, kernel)
        
        segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels_im = cv2.connectedComponents(segmented_img)
        if num_labels > 1: 
            sizes = []
            for label in range(1, num_labels):
                sizes.append(np.sum(labels_im == label))
            if sizes:
                min_size = max(sizes) * 0.05  
                for label in range(1, num_labels):
                    if np.sum(labels_im == label) < min_size:
                        segmented_img[labels_im == label] = 0
    
    return segmented_img

def visualize_segmentation(original, enhanced, segmented, filename):
    """Create side-by-side visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Enhanced')
    axes[1].axis('off')

    axes[2].imshow(enhanced, cmap='gray')
    axes[2].imshow(segmented, cmap='jet', alpha=0.5)
    axes[2].set_title('Segmentation Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_ROOT, f"vis_{filename}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_dataset():
    """Process all images in wavelet_enhanced folder"""
    count = 0
    processed_files = []
    
    for root, _, files in os.walk(INPUT_ROOT):
        for f in tqdm(files, desc="Processing images"):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):

                src = os.path.join(root, f)
                rel_path = os.path.relpath(src, INPUT_ROOT)
                
                dst = os.path.join(OUTPUT_ROOT, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                enhanced_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if enhanced_img is None:
                    print(f"⚠️ Could not read: {src}")
                    continue
                
                segmented_mask = kmeans_segmentation(enhanced_img, K=3, post_process=True)
                
                cv2.imwrite(dst, segmented_mask)
                
                original_path = src.replace('wavelet_enhanced', 'preprocessed_anisotropic')
                original_img = None
                if os.path.exists(original_path):
                    original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

                visualize_segmentation(original_img, enhanced_img, segmented_mask, f)
                
                count += 1
                processed_files.append(f)
    
    return count, processed_files

if __name__ == "__main__":
    print("=" * 60)
    print("K-MEANS SEGMENTATION PIPELINE")
    print("=" * 60)
    print(f"Input directory: {INPUT_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Visualization directory: {VISUALIZATION_ROOT}")

    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Input directory not found: {INPUT_ROOT}")
        print("Please ensure wavelet_enhanced folder exists with images.")
    else:
        image_count = sum([len(files) for r, d, files in os.walk(INPUT_ROOT)])
        print(f"📁 Found approximately {image_count} images to process")
        
        count, files = process_dataset()
        
        print(f"\n✅ K-means segmentation completed for {count} images")
        print(f"📊 Segmentation masks saved to: {OUTPUT_ROOT}")
        print(f"🎨 Visualizations saved to: {VISUALIZATION_ROOT}")
        
        if files:
            print("\n📋 Sample of processed files:")
            for f in files[:5]:
                print(f"  - {f}")
            if len(files) > 5:
                print(f"  ... and {len(files)-5} more")

def test_single_image(image_path, K=3):
    """Test K-means on a single image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    print(f"📐 Image shape: {img.shape}")
    print(f"📊 Intensity range: {img.min()} - {img.max()}")
    
    segmented = kmeans_segmentation(img, K=K)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Input Image\nMin: {img.min()}, Max: {img.max()}')
    axes[0].axis('off')
    
    axes[1].imshow(segmented, cmap='gray')
    axes[1].set_title(f'Segmented Mask (K={K})')
    axes[1].axis('off')
    
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(segmented, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    test_output = "/content/test_kmeans_result.png"
    fig.savefig(test_output, dpi=150, bbox_inches='tight')
    print(f"💾 Test visualization saved to: {test_output}")
    
    tumor_pixels = np.sum(segmented > 0)
    total_pixels = img.shape[0] * img.shape[1]
    tumor_percentage = (tumor_pixels / total_pixels) * 100
    
    print(f"📈 Tumor region: {tumor_pixels} pixels ({tumor_percentage:.2f}% of image)")
