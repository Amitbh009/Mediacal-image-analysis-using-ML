import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# ================= IMAGE COMPARISON SCRIPT =================
# Compare original vs anisotropic diffusion filtered images

# -------- PATHS --------
INPUT_ROOT = "/kaggle/input/brain-tumor-mri-dataset"
OUTPUT_ROOT = "/content/drive/MyDrive/MyProject/preprocessed_anisotropic"

# -------- GET SAMPLE IMAGES --------
def get_sample_images(num_samples=5):
    """Get random sample of original and processed image pairs"""
    samples = []
    
    # Get all processed images
    processed_images = []
    for root, _, files in os.walk(OUTPUT_ROOT):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                processed_path = os.path.join(root, f)
                # Find corresponding original
                rel_path = os.path.relpath(processed_path, OUTPUT_ROOT)
                original_path = os.path.join(INPUT_ROOT, rel_path)
                
                if os.path.exists(original_path):
                    processed_images.append((original_path, processed_path))
    
    # Select random samples
    if len(processed_images) > num_samples:
        samples = random.sample(processed_images, num_samples)
    else:
        samples = processed_images
    
    return samples

# -------- DISPLAY COMPARISON --------
def display_comparison(original_path, processed_path):
    """Display original and processed images side by side"""
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        print(f"Could not load one of the images")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original\n' + os.path.basename(original_path))
    axes[0].axis('off')
    
    # Processed image
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('Anisotropic Diffusion\n' + os.path.basename(processed_path))
    axes[1].axis('off')
    
    # Difference image
    diff = cv2.absdiff(original, processed)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference (max diff: {diff.max()})')
    axes[2].axis('off')
    
    # Stats
    print(f"📊 Image: {os.path.basename(original_path)}")
    print(f"   Original shape: {original.shape}")
    print(f"   Original range: [{original.min()}, {original.max()}]")
    print(f"   Processed range: [{processed.min()}, {processed.max()}]")
    print(f"   Mean difference: {diff.mean():.2f}")
    print(f"   Max difference: {diff.max()}")
    print("-" * 50)
    
    plt.tight_layout()
    plt.show()

# -------- QUICK STATISTICAL ANALYSIS --------
def analyze_changes():
    """Analyze overall changes between original and processed images"""
    print("🔍 Analyzing preprocessing effects...")
    
    all_differences = []
    all_original_means = []
    all_processed_means = []
    
    # Get all pairs
    pairs = []
    for root, _, files in os.walk(OUTPUT_ROOT):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                processed_path = os.path.join(root, f)
                rel_path = os.path.relpath(processed_path, OUTPUT_ROOT)
                original_path = os.path.join(INPUT_ROOT, rel_path)
                
                if os.path.exists(original_path):
                    pairs.append((original_path, processed_path))
    
    print(f"Found {len(pairs)} image pairs to analyze")
    
    # Analyze first 100 images for speed (or adjust as needed)
    sample_size = min(100, len(pairs))
    sample_pairs = random.sample(pairs, sample_size) if len(pairs) > 100 else pairs
    
    for orig_path, proc_path in sample_pairs:
        original = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        processed = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)
        
        if original is not None and processed is not None:
            diff = cv2.absdiff(original, processed)
            all_differences.append(diff.mean())
            all_original_means.append(original.mean())
            all_processed_means.append(processed.mean())
    
    if all_differences:
        print("\n📈 STATISTICAL SUMMARY:")
        print(f"   Images analyzed: {len(all_differences)}")
        print(f"   Average mean difference: {np.mean(all_differences):.2f}")
        print(f"   Max mean difference: {np.max(all_differences):.2f}")
        print(f"   Original mean intensity: {np.mean(all_original_means):.2f}")
        print(f"   Processed mean intensity: {np.mean(all_processed_means):.2f}")
        print(f"   Intensity change: {np.mean(all_processed_means) - np.mean(all_original_means):.2f}")
    
    return all_differences

# -------- MAIN EXECUTION --------
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 COMPARING ORIGINAL vs DENOISED IMAGES")
    print("=" * 60)
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_ROOT):
        print(f"❌ Output directory not found: {OUTPUT_ROOT}")
        print("Please run the preprocessing script first.")
    else:
        # Get and display random samples
        samples = get_sample_images(num_samples=3)
        
        if not samples:
            print("❌ No processed images found!")
        else:
            print(f"✅ Found {len(samples)} image pairs to compare")
            
            # Display each sample
            for orig_path, proc_path in samples:
                display_comparison(orig_path, proc_path)
            
            # Run statistical analysis
            analyze_changes()
            
            print("\n✅ Comparison complete!")
            print("\n💡 Observations:")
            print("   - Original: May have more noise/speckles")
            print("   - Processed: Smoother homogeneous regions")
            print("   - Processed: Preserved/emphasized edges")
            print("   - Difference image shows removed noise")