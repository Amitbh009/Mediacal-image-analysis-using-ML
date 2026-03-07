import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy import stats
from pathlib import Path

# -------- PATHS --------
SEGMENTED_ROOT = "/content/drive/MyDrive/segment_data/kmeans_segmented"           
ORIGINAL_ROOT = "/content/drive/MyDrive/preprocessed_anisotropic"   
ENHANCED_ROOT = "/content/drive/MyDrive/wavelet_enhanced"        
FEATURES_ROOT = "/content/features_complete"       

os.makedirs(FEATURES_ROOT, exist_ok=True)

def find_matching_image(mask_path, search_root):
    """Find matching image in another directory"""
    rel_path = os.path.relpath(mask_path, SEGMENTED_ROOT)
    possible_paths = [
        os.path.join(search_root, rel_path),
        os.path.join(search_root, os.path.basename(mask_path))
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(search_root):
        if os.path.basename(mask_path) in files:
            return os.path.join(root, os.path.basename(mask_path))
    
    return None

def resize_to_match(source_img, target_shape, is_mask=False):
    """Resize source image to match target shape"""
    if source_img.shape == target_shape:
        return source_img.copy()
    
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    resized = cv2.resize(source_img, (target_shape[1], target_shape[0]), 
                        interpolation=interpolation)
    return resized

def load_and_match_images(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None, None
    

    original_path = find_matching_image(mask_path, ORIGINAL_ROOT)
    original = None
    if original_path and os.path.exists(original_path):
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if original is not None and original.shape != mask.shape:
            original = resize_to_match(original, mask.shape, is_mask=False)
    
    enhanced_path = find_matching_image(mask_path, ENHANCED_ROOT)
    enhanced = None
    if enhanced_path and os.path.exists(enhanced_path):
        enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
        if enhanced is not None and enhanced.shape != mask.shape:
            enhanced = resize_to_match(enhanced, mask.shape, is_mask=False)
    
    if original is None:
        original = enhanced if enhanced is not None else mask
    
    return mask, original, enhanced

def extract_lbp_features_safe(image, radius=3, n_points=24):
    """Safe LBP feature extraction"""
    try:
        image_ubyte = img_as_ubyte(image)
        
        if len(image_ubyte.shape) > 2:
            image_ubyte = cv2.cvtColor(image_ubyte, cv2.COLOR_BGR2GRAY)
        
        lbp = local_binary_pattern(image_ubyte, n_points, radius, 'uniform')
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        features = {
            'lbp_mean': np.mean(hist),
            'lbp_std': np.std(hist),
            'lbp_skewness': stats.skew(hist),
            'lbp_kurtosis': stats.kurtosis(hist),
            'lbp_entropy': stats.entropy(hist + 1e-10),
            'lbp_energy': np.sum(hist**2),
            'lbp_uniformity': np.max(hist),
            'lbp_contrast': np.std(lbp)
        }
        
        for i in range(min(10, len(hist))):
            features[f'lbp_hist_bin_{i}'] = hist[i]
        
        return features, lbp
    except Exception as e:
        default_features = {f'lbp_{k}': 0 for k in ['mean', 'std', 'skewness', 'kurtosis', 
                                                   'entropy', 'energy', 'uniformity', 'contrast']}
        for i in range(10):
            default_features[f'lbp_hist_bin_{i}'] = 0
        return default_features, np.zeros_like(image)

def extract_glcm_features_safe(image, distances=[1, 3, 5]):
    try:
        image_ubyte = img_as_ubyte(image)
        if len(image_ubyte.shape) > 2:
            image_ubyte = cv2.cvtColor(image_ubyte, cv2.COLOR_BGR2GRAY)
        
        levels = min(16, np.max(image_ubyte) + 1)
        if levels < 2:
            levels = 8
        image_quantized = (image_ubyte // (256 // levels)).astype(np.uint8)
        
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image_quantized, 
                          distances=distances, 
                          angles=angles,
                          levels=levels,
                          symmetric=True,
                          normed=True)
        
        features = {}
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in glcm_props:
            prop_values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(prop_values)
            features[f'glcm_{prop}_std'] = np.std(prop_values)
            features[f'glcm_{prop}_max'] = np.max(prop_values)
            features[f'glcm_{prop}_min'] = np.min(prop_values)
        
        features['glcm_entropy'] = -np.sum(glcm * np.log2(glcm + 1e-10))
        features['glcm_variance'] = np.var(glcm)
        
        return features, glcm
    except Exception as e:
        default_features = {}
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in glcm_props:
            default_features.update({f'glcm_{prop}_{stat}': 0 for stat in ['mean', 'std', 'max', 'min']})
        default_features['glcm_entropy'] = 0
        default_features['glcm_variance'] = 0
        return default_features, None

def extract_shape_features_safe(mask):
    try:
        mask_binary = (mask > 0).astype(np.uint8)
        
        features = {}
        
        features['mask_area'] = np.sum(mask_binary)
        features['mask_percentage'] = (features['mask_area'] / mask.size) * 100
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            features['shape_area'] = area
            features['shape_perimeter'] = perimeter
            features['shape_compactness'] = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
            features['shape_circularity'] = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            features['shape_aspect_ratio'] = w / h if h > 0 else 0
            features['shape_rectangularity'] = area / (w * h) if (w * h) > 0 else 0
            
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                for i in range(7):
                    hu_val = hu_moments[i]
                    if hu_val != 0:
                        features[f'shape_hu_{i+1}'] = -np.sign(hu_val) * np.log10(np.abs(hu_val) + 1e-10)
                    else:
                        features[f'shape_hu_{i+1}'] = 0
            else:
                for i in range(7):
                    features[f'shape_hu_{i+1}'] = 0
        else:
            for i in range(7):
                features[f'shape_hu_{i+1}'] = 0
            features.update({
                'shape_area': 0, 'shape_perimeter': 0, 'shape_compactness': 0,
                'shape_circularity': 0, 'shape_aspect_ratio': 0, 'shape_rectangularity': 0
            })
        
        return features
    except Exception as e:
        default_features = {f'shape_hu_{i+1}': 0 for i in range(7)}
        default_features.update({
            'mask_area': 0, 'mask_percentage': 0,
            'shape_area': 0, 'shape_perimeter': 0, 'shape_compactness': 0,
            'shape_circularity': 0, 'shape_aspect_ratio': 0, 'shape_rectangularity': 0
        })
        return default_features

def extract_intensity_features_safe(image, mask):
    """Safe intensity feature extraction"""
    try:
        mask_binary = (mask > 0).astype(np.uint8)
        
        masked_region = image[mask_binary > 0]
        
        features = {}
        
        if len(masked_region) > 10:
            features['intensity_mean'] = np.mean(masked_region)
            features['intensity_std'] = np.std(masked_region)
            features['intensity_skewness'] = stats.skew(masked_region)
            features['intensity_kurtosis'] = stats.kurtosis(masked_region)
            features['intensity_median'] = np.median(masked_region)
            features['intensity_min'] = np.min(masked_region)
            features['intensity_max'] = np.max(masked_region)
            features['intensity_range'] = features['intensity_max'] - features['intensity_min']
            features['intensity_iqr'] = np.percentile(masked_region, 75) - np.percentile(masked_region, 25)
            
    
            hist, _ = np.histogram(masked_region, bins=20, density=True)
            features['intensity_entropy'] = stats.entropy(hist + 1e-10)
        else:
            intensity_features = ['mean', 'std', 'skewness', 'kurtosis', 'median', 
                                'min', 'max', 'range', 'iqr', 'entropy']
            for feat in intensity_features:
                features[f'intensity_{feat}'] = 0
        
        return features
    except Exception as e:
        intensity_features = ['mean', 'std', 'skewness', 'kurtosis', 'median', 
                            'min', 'max', 'range', 'iqr', 'entropy']
        return {f'intensity_{feat}': 0 for feat in intensity_features}

def extract_all_features_safe(mask, original, enhanced):
    texture_image = enhanced if enhanced is not None else original
    
    lbp_features, lbp_img = extract_lbp_features_safe(texture_image)
    glcm_features, glcm_mat = extract_glcm_features_safe(texture_image)
    shape_features = extract_shape_features_safe(mask)
    intensity_features = extract_intensity_features_safe(original, mask)
    
    all_features = {}
    all_features.update(lbp_features)
    all_features.update(glcm_features)
    all_features.update(shape_features)
    all_features.update(intensity_features)
    
    try:
        all_features['texture_complexity'] = all_features.get('lbp_entropy', 0) * all_features.get('glcm_entropy', 0)
        all_features['heterogeneity_score'] = all_features.get('intensity_std', 0) * all_features.get('glcm_contrast_mean', 0)
        all_features['regularity_index'] = all_features.get('lbp_uniformity', 0) * all_features.get('glcm_homogeneity_mean', 0)
    except:
        all_features['texture_complexity'] = 0
        all_features['heterogeneity_score'] = 0
        all_features['regularity_index'] = 0
    
    return all_features

def process_complete_dataset(sample_size=None):
    print("=" * 80)
    print("COMPLETE FEATURE EXTRACTION WITH SIZE MATCHING")
    print("=" * 80)
    
    mask_paths = []
    for root, dirs, files in os.walk(SEGMENTED_ROOT):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                mask_paths.append(os.path.join(root, file))
    
    if sample_size:
        mask_paths = mask_paths[:sample_size]
    
    print(f"Found {len(mask_paths)} mask images")
    
    all_features = []
    failed_images = []
    success_count = 0
    
    for mask_path in tqdm(mask_paths, desc="Processing images"):
        try:
            mask, original, enhanced = load_and_match_images(mask_path)
            
            if mask is None:
                failed_images.append((os.path.basename(mask_path), "Failed to load mask"))
                continue
            
            if original is None:
                failed_images.append((os.path.basename(mask_path), "No matching original/enhanced found"))
                continue
            
            features = extract_all_features_safe(mask, original, enhanced)
            
            features['image_name'] = os.path.basename(mask_path)
            features['image_path'] = mask_path
            features['mask_shape'] = f"{mask.shape[0]}x{mask.shape[1]}"
            features['original_found'] = original is not None
            features['enhanced_found'] = enhanced is not None
            
            all_features.append(features)
            success_count += 1
            
        except Exception as e:
            failed_images.append((os.path.basename(mask_path), str(e)))
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully processed: {success_count} images")
        print(f"❌ Failed: {len(failed_images)} images")
        
        if failed_images:
            print("\nFirst 10 failed images:")
            for img, error in failed_images[:10]:
                print(f"  - {img}: {error}")
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(FEATURES_ROOT, f"features_complete_{timestamp}.csv")
        pkl_path = os.path.join(FEATURES_ROOT, f"features_complete_{timestamp}.pkl")
        
        features_df.to_csv(csv_path, index=False)
        features_df.to_pickle(pkl_path)
        
        print(f"\n💾 Features saved to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - Pickle: {pkl_path}")
        
        return features_df, failed_images
    else:
        print("❌ No features extracted!")
        return None, failed_images

def process_all_images():
    """Process ALL images"""
    print("=" * 80)
    print("PROCESSING ALL IMAGES")
    print("=" * 80)
    
    total_images = 0
    for root, dirs, files in os.walk(SEGMENTED_ROOT):
        total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Total images to process: {total_images}")
    print("Estimated time: 60-90 minutes")
    
    batch_size = 1000
    all_batches_df = []
    
    all_paths = []
    for root, dirs, files in os.walk(SEGMENTED_ROOT):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                all_paths.append(os.path.join(root, file))
    
    total_batches = (len(all_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_paths))
        batch_paths = all_paths[start_idx:end_idx]
        
        print(f"\n📦 Processing batch {batch_idx+1}/{total_batches} (images {start_idx+1}-{end_idx})")
        
        batch_features = []
        batch_failed = []
        
        for mask_path in tqdm(batch_paths, desc=f"Batch {batch_idx+1}"):
            try:
                mask, original, enhanced = load_and_match_images(mask_path)
                
                if mask is not None and original is not None:
                    features = extract_all_features_safe(mask, original, enhanced)
                    features['image_name'] = os.path.basename(mask_path)
                    features['image_path'] = mask_path
                    batch_features.append(features)
                else:
                    batch_failed.append(os.path.basename(mask_path))
            except:
                batch_failed.append(os.path.basename(mask_path))
        
        if batch_features:
            batch_df = pd.DataFrame(batch_features)
            all_batches_df.append(batch_df)
            
            batch_csv = os.path.join(FEATURES_ROOT, f"features_batch_{batch_idx+1}.csv")
            batch_df.to_csv(batch_csv, index=False)
            print(f"  ✅ Saved batch {batch_idx+1}: {len(batch_df)} images")
        
        print(f"  ❌ Failed in batch: {len(batch_failed)} images")
    
    if all_batches_df:
        final_df = pd.concat(all_batches_df, ignore_index=True)
        
        final_csv = os.path.join(FEATURES_ROOT, "features_ALL_complete.csv")
        final_pkl = os.path.join(FEATURES_ROOT, "features_ALL_complete.pkl")
        
        final_df.to_csv(final_csv, index=False)
        final_df.to_pickle(final_pkl)
        
        print("\n" + "=" * 80)
        print("ALL IMAGES PROCESSED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Total processed: {len(final_df)} images")
        print(f"💾 Combined features saved to:")
        print(f"  - {final_csv}")
        print(f"  - {final_pkl}")
        
        return final_df
    else:
        print("❌ No images were processed successfully!")
        return None

def fix_size_issues_and_reprocess():
    """Fix size mismatches and reprocess failed images"""
    print("=" * 80)
    print("FIXING SIZE MISMATCHES AND REPROCESSING")
    print("=" * 80)
    
    fixed_dir = "/content/kmeans_segmented_fixed"
    os.makedirs(fixed_dir, exist_ok=True)
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(SEGMENTED_ROOT):
        for file in tqdm(files, desc="Checking sizes"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    continue
                
                rel_path = os.path.relpath(mask_path, SEGMENTED_ROOT)
                orig_path = os.path.join(ORIGINAL_ROOT, rel_path)
                
                if os.path.exists(orig_path):
                    original = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
                    if original is not None and original.shape != mask.shape:
                        mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                        
                        fixed_path = os.path.join(fixed_dir, rel_path)
                        os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
                        cv2.imwrite(fixed_path, mask_resized)
                        fixed_count += 1
                    else:
                        fixed_path = os.path.join(fixed_dir, rel_path)
                        os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
                        cv2.imwrite(fixed_path, mask)
                else:
                    fixed_path = os.path.join(fixed_dir, rel_path)
                    os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
                    cv2.imwrite(fixed_path, mask)
    
    print(f"\n✅ Fixed {fixed_count} size mismatches")
    print(f"✅ Fixed masks saved to: {fixed_dir}")
    
    return fixed_dir

def quick_fix_and_process():
    global SEGMENTED_ROOT 
    
    print("=" * 80)
    print("QUICK FIX AND PROCESS ALL IMAGES")
    print("=" * 80)
    
    print("\nStep 1: Fixing size mismatches...")
    fixed_dir = fix_size_issues_and_reprocess()
    
    original_root = SEGMENTED_ROOT
    SEGMENTED_ROOT = fixed_dir
    
    print(f"\nStep 2: Updated SEGMENTED_ROOT from {original_root} to {SEGMENTED_ROOT}")
    
    print("\nStep 3: Processing all images...")
    final_df = process_all_images()
    
    if final_df is not None:
        print("\n✅ All images processed successfully!")
        print(f"   Processed: {len(final_df)} images")
        print(f"   Features: {final_df.shape[1] - 5}")
        print(f"   Saved to: /content/features_complete/")
    
    SEGMENTED_ROOT = original_root
    
    return final_df

def main():
    os.makedirs(FEATURES_ROOT, exist_ok=True)
    
    print("=" * 80)
    print("COMPLETE FEATURE EXTRACTION FOR ALL IMAGES")
    print("=" * 80)
    
    print("\nOption 1: Test with 50 images")
    test_df, test_failed = process_complete_dataset(sample_size=50)
    
    if test_df is not None:
        print(f"\n✅ Test successful! Processed {len(test_df)} images")
        
        print("\n" + "=" * 80)
        print("RECOMMENDED: Fix size issues before processing all images")
        print("=" * 80)
        
        fix_response = input("Fix size mismatches first? (yes/no): ")
        
        if fix_response.lower() in ['yes', 'y']:
            fixed_dir = fix_size_issues_and_reprocess()
            
            global SEGMENTED_ROOT
            SEGMENTED_ROOT = fixed_dir
            
            print(f"\n✅ Updated SEGMENTED_ROOT to: {SEGMENTED_ROOT}")
        
        print("\n" + "=" * 80)
        process_response = input("Process ALL images? (yes/no): ")
        
        if process_response.lower() in ['yes', 'y']:
            final_df = process_all_images()
            
            if final_df is not None:
                print("\n" + "=" * 80)
                print("FEATURE EXTRACTION SUMMARY")
                print("=" * 80)
                print(f"Total images: {len(final_df)}")
                print(f"Total features: {final_df.shape[1] - 5}") 
                print(f"Feature columns: {list(final_df.columns[:15])}...")
                
                stats = final_df.describe()
                stats_csv = os.path.join(FEATURES_ROOT, "feature_statistics_all.csv")
                stats.to_csv(stats_csv)
                print(f"\n📊 Statistics saved to: {stats_csv}")
        
        else:
            print("\nProcessing complete! Run process_all_images() when ready.")
    
    else:
        print("❌ Test failed! Check the error messages above.")

if __name__ == "__main__":
    main()