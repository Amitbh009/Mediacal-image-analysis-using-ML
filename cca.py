import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class CCAAnalyzer:
    def __init__(self, base_path="segmented_output"):
        """
        Initialize CCA Analyzer for tumor segmentation masks
        
        Args:
            base_path: Path to your segmented_output directory
        """
        self.base_path = Path(base_path)
        self.results = []
        self.processed_counts = {'glioma': 0, 'meningioma': 0, 'notumor': 0, 'pituitary': 0}
        
    def find_mask_files(self, verbose=True):
        """
        Find all mask files in the Training directory across all tumor types
        
        Returns:
            List of dictionaries with mask file information
        """
        mask_files = []
        
        # Define tumor types
        tumor_types = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        for tumor_type in tumor_types:
            # Try multiple possible paths
            possible_paths = [
                self.base_path / "Training" / tumor_type / "masks",
                self.base_path / tumor_type / "masks",
                self.base_path / "masks" / tumor_type,
                self.base_path / tumor_type  # direct folder
            ]
            
            mask_dir = None
            for path in possible_paths:
                if path.exists():
                    mask_dir = path
                    if verbose:
                        print(f"Found {tumor_type} masks at: {path}")
                    break
            
            if mask_dir is None:
                if verbose:
                    print(f"Warning: Could not find masks for {tumor_type}")
                continue
            
            # Count files found
            file_count = 0
            
            # Look for various image formats
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            
            for ext in image_extensions:
                files = list(mask_dir.glob(ext))
                file_count += len(files)
                
                for mask_file in files:
                    mask_files.append({
                        'path': mask_file,
                        'type': tumor_type,
                        'filename': mask_file.name,
                        'full_path': str(mask_file)
                    })
            
            self.processed_counts[tumor_type] = file_count
            
            if verbose:
                print(f"  Found {file_count} mask files for {tumor_type}")
        
        if verbose:
            print(f"\nTotal mask files found: {len(mask_files)}")
            for tumor_type, count in self.processed_counts.items():
                print(f"  {tumor_type}: {count} files")
        
        return mask_files
    
    def verify_directory_structure(self):
        """Check and print the actual directory structure"""
        print("=" * 60)
        print("CHECKING DIRECTORY STRUCTURE")
        print("=" * 60)
        
        base = Path(self.base_path)
        if not base.exists():
            print(f"ERROR: Base path does not exist: {base}")
            return False
        
        print(f"Base path: {base}")
        print(f"Exists: {base.exists()}")
        print(f"Is directory: {base.is_dir()}")
        
        print("\nContents of base directory:")
        for item in base.iterdir():
            print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
        
        # Check Training directory
        training_dir = base / "Training"
        if training_dir.exists():
            print(f"\nTraining directory exists: {training_dir}")
            print("Contents of Training directory:")
            for item in training_dir.iterdir():
                print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
                
                # Check each tumor type directory
                if item.is_dir():
                    tumor_path = item
                    print(f"    Contents of {item.name}:")
                    for subitem in tumor_path.iterdir():
                        print(f"      - {subitem.name} ({'DIR' if subitem.is_dir() else 'FILE'})")
        else:
            print(f"\nTraining directory NOT found: {training_dir}")
            
            # Check if tumor folders are directly in base
            print("\nChecking for tumor folders directly in base directory:")
            tumor_types = ['glioma', 'meningioma', 'notumor', 'pituitary']
            for tumor_type in tumor_types:
                tumor_dir = base / tumor_type
                if tumor_dir.exists():
                    print(f"  ✓ Found {tumor_type} directory")
                    print(f"    Contents of {tumor_type}:")
                    for subitem in tumor_dir.iterdir():
                        print(f"      - {subitem.name} ({'DIR' if subitem.is_dir() else 'FILE'})")
                else:
                    print(f"  ✗ {tumor_type} directory not found")
        
        return True
    
    def apply_cca(self, mask_path, min_size=100, connectivity=8):
        """
        Apply Connected Components Analysis to a mask
        
        Args:
            mask_path: Path to the mask image
            min_size: Minimum component size (pixels)
            connectivity: 4 or 8 connectivity for component labeling
        
        Returns:
            Dictionary with CCA results or None if error
        """
        try:
            # Read mask image
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                return None
            
            # Debug: Print mask info
            # print(f"Processing {mask_path.name}: Shape={mask.shape}, Max={mask.max()}, Min={mask.min()}")
            
            # Ensure binary mask (threshold if needed)
            if mask.max() > 1:
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                binary_mask = (mask * 255).astype(np.uint8)
            
            # Check if mask is empty
            if np.sum(binary_mask) == 0:
                # print(f"Warning: Empty mask for {mask_path.name}")
                return {
                    'num_components': 0,
                    'components': [],
                    'labels_map': np.zeros_like(binary_mask),
                    'original_mask': binary_mask,
                    'stats': None
                }
            
            # Apply CCA using OpenCV
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=connectivity, ltype=cv2.CV_32S
            )
            
            # Filter small components
            filtered_labels = labels.copy()
            for label in range(1, num_labels):  # Skip background (label 0)
                if stats[label, cv2.CC_STAT_AREA] < min_size:
                    filtered_labels[filtered_labels == label] = 0
            
            # Calculate component properties using skimage
            bool_mask = filtered_labels > 0
            if np.any(bool_mask):
                labeled_mask = measure.label(bool_mask, connectivity=1)
                properties = measure.regionprops(labeled_mask)
            else:
                properties = []
            
            # Prepare results
            components_info = []
            for i, prop in enumerate(properties):
                component_data = {
                    'component_id': i + 1,
                    'area': prop.area,
                    'perimeter': prop.perimeter,
                    'centroid': prop.centroid,
                    'bbox': prop.bbox,
                    'major_axis_length': prop.major_axis_length,
                    'minor_axis_length': prop.minor_axis_length,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'extent': prop.extent,
                    'equivalent_diameter': prop.equivalent_diameter
                }
                components_info.append(component_data)
            
            return {
                'num_components': len(components_info),
                'components': components_info,
                'labels_map': filtered_labels,
                'original_mask': binary_mask,
                'stats': stats
            }
            
        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
            return None
    
    def analyze_all_masks(self, min_size=100, max_samples_per_type=None, verbose=True):
        """
        Apply CCA to all mask files and collect statistics
        
        Args:
            min_size: Minimum component size in pixels
            max_samples_per_type: Maximum number of samples per tumor type (for testing)
            verbose: Print progress information
        """
        # First verify directory structure
        self.verify_directory_structure()
        
        # Find mask files
        mask_files = self.find_mask_files(verbose=verbose)
        
        if not mask_files:
            print("ERROR: No mask files found!")
            return
        
        # Reset results
        self.results = []
        stats_by_type = {t: {'count': 0, 'errors': 0} for t in ['glioma', 'meningioma', 'notumor', 'pituitary']}
        
        if verbose:
            print(f"\nStarting CCA analysis on {len(mask_files)} mask files...")
            print("=" * 60)
        
        # Process each mask file
        for mask_info in tqdm(mask_files, desc="Processing masks"):
            tumor_type = mask_info['type']
            
            # Skip if we've reached max samples for this type
            if (max_samples_per_type is not None and 
                stats_by_type[tumor_type]['count'] >= max_samples_per_type):
                continue
            
            # Apply CCA
            cca_result = self.apply_cca(mask_info['path'], min_size=min_size)
            
            if cca_result is None:
                stats_by_type[tumor_type]['errors'] += 1
                continue
            
            # Calculate areas
            areas = [c['area'] for c in cca_result['components']]
            
            # Store results
            self.results.append({
                'filename': mask_info['filename'],
                'tumor_type': tumor_type,
                'num_components': cca_result['num_components'],
                'total_area': sum(areas) if areas else 0,
                'largest_area': max(areas) if areas else 0,
                'avg_area': np.mean(areas) if areas else 0,
                'std_area': np.std(areas) if areas else 0,
                'components': cca_result['components'],
                'path': mask_info['full_path']
            })
            
            stats_by_type[tumor_type]['count'] += 1
        
        # Print statistics
        if verbose:
            print("\n" + "=" * 60)
            print("PROCESSING STATISTICS")
            print("=" * 60)
            for tumor_type, stats in stats_by_type.items():
                if stats['count'] > 0 or stats['errors'] > 0:
                    print(f"{tumor_type.upper():12} - Processed: {stats['count']:4d}, Errors: {stats['errors']:3d}")
            
            print(f"\nTotal processed successfully: {len(self.results)}")
            
            # Count by tumor type
            tumor_counts = {}
            for result in self.results:
                tumor_type = result['tumor_type']
                tumor_counts[tumor_type] = tumor_counts.get(tumor_type, 0) + 1
            
            print("\nResults by tumor type:")
            for tumor_type, count in sorted(tumor_counts.items()):
                print(f"  {tumor_type}: {count} samples")
    
    def export_results(self, output_dir="cca_results"):
        """
        Export CCA results to CSV and images
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not self.results:
            print("No results to export!")
            return
        
        print(f"\nExporting results to {output_dir}...")
        
        # Save summary statistics
        summary_data = []
        for result in self.results:
            summary_data.append({
                'filename': result['filename'],
                'tumor_type': result['tumor_type'],
                'num_components': result['num_components'],
                'total_area': result['total_area'],
                'largest_area': result['largest_area'],
                'avg_area': result['avg_area'],
                'std_area': result['std_area'],
                'file_path': result.get('path', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_dir / "cca_summary_all.csv", index=False)
        print(f"Summary saved to {output_dir}/cca_summary_all.csv")
        
        # Also save separate CSV files for each tumor type
        for tumor_type in df_summary['tumor_type'].unique():
            type_df = df_summary[df_summary['tumor_type'] == tumor_type]
            type_df.to_csv(output_dir / f"cca_summary_{tumor_type}.csv", index=False)
            print(f"  {tumor_type}: {len(type_df)} samples")
        
        # Create detailed results
        detailed_data = []
        for result in self.results:
            for component in result['components']:
                detailed_data.append({
                    'filename': result['filename'],
                    'tumor_type': result['tumor_type'],
                    'component_id': component['component_id'],
                    'area': component['area'],
                    'perimeter': component['perimeter'],
                    'major_axis_length': component['major_axis_length'],
                    'minor_axis_length': component['minor_axis_length'],
                    'eccentricity': component['eccentricity'],
                    'solidity': component['solidity'],
                    'extent': component['extent'],
                    'equivalent_diameter': component['equivalent_diameter']
                })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_csv(output_dir / "cca_detailed_all.csv", index=False)
            print(f"Detailed results saved to {output_dir}/cca_detailed_all.csv")
        
        # Save aggregated statistics
        self.save_aggregated_stats(output_dir)
        
        # Save sample visualizations
        self.save_sample_visualizations(output_dir, num_samples=3)
        
        print(f"\nAll results exported to {output_dir}")
    
    def save_aggregated_stats(self, output_dir):
        """Save aggregated statistics by tumor type"""
        if not self.results:
            return
        
        aggregated_stats = []
        for tumor_type in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            type_results = [r for r in self.results if r['tumor_type'] == tumor_type]
            
            if not type_results:
                continue
            
            areas = [r['total_area'] for r in type_results]
            components = [r['num_components'] for r in type_results]
            largest_areas = [r['largest_area'] for r in type_results]
            
            # Calculate fragmentation rate (percentage with >1 component)
            fragmented = sum(1 for r in type_results if r['num_components'] > 1)
            fragmentation_rate = (fragmented / len(type_results)) * 100 if type_results else 0
            
            aggregated_stats.append({
                'tumor_type': tumor_type,
                'num_samples': len(type_results),
                'avg_components': np.mean(components),
                'std_components': np.std(components),
                'median_components': np.median(components),
                'avg_total_area': np.mean(areas),
                'std_total_area': np.std(areas),
                'median_total_area': np.median(areas),
                'avg_largest_area': np.mean(largest_areas),
                'fragmentation_rate': fragmentation_rate,
                'min_area': np.min(areas) if areas else 0,
                'max_area': np.max(areas) if areas else 0
            })
        
        df_aggregated = pd.DataFrame(aggregated_stats)
        df_aggregated.to_csv(output_dir / "aggregated_stats_all.csv", index=False)
        print(f"Aggregated statistics saved to {output_dir}/aggregated_stats_all.csv")
        
        # Print summary
        print("\n" + "=" * 60)
        print("AGGREGATED STATISTICS")
        print("=" * 60)
        for stats in aggregated_stats:
            print(f"\n{stats['tumor_type'].upper()}:")
            print(f"  Samples: {stats['num_samples']}")
            print(f"  Avg Components: {stats['avg_components']:.2f} ± {stats['std_components']:.2f}")
            print(f"  Avg Area: {stats['avg_total_area']:.0f} ± {stats['std_total_area']:.0f} px²")
            print(f"  Fragmentation Rate: {stats['fragmentation_rate']:.1f}%")
    
    def save_sample_visualizations(self, output_dir, num_samples=3):
        """Save visualizations for sample images from each tumor type"""
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        for tumor_type in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            # Get samples of this tumor type
            type_samples = [r for r in self.results if r['tumor_type'] == tumor_type]
            
            if not type_samples:
                print(f"No samples found for {tumor_type}")
                continue
            
            # Take first few samples
            for i, sample in enumerate(type_samples[:num_samples]):
                # Find the original mask file
                mask_path = sample.get('path', '')
                if not mask_path or not Path(mask_path).exists():
                    continue
                
                cca_result = self.apply_cca(mask_path, min_size=50)
                
                if cca_result:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original mask
                    axes[0].imshow(cca_result['original_mask'], cmap='gray')
                    axes[0].set_title(f"Original: {tumor_type}")
                    axes[0].axis('off')
                    
                    # CCA result
                    labeled_img = cca_result['labels_map']
                    if labeled_img.max() > 0:
                        norm_img = labeled_img / labeled_img.max()
                        axes[1].imshow(norm_img, cmap='tab20')
                    else:
                        axes[1].imshow(labeled_img, cmap='gray')
                    axes[1].set_title(f"Components: {cca_result['num_components']}")
                    axes[1].axis('off')
                    
                    # Overlay
                    if cca_result['num_components'] > 0:
                        colored_labels = np.zeros((*labeled_img.shape, 3), dtype=np.uint8)
                        for label in range(1, labeled_img.max() + 1):
                            colored_labels[labeled_img == label] = np.random.randint(50, 255, 3)
                        
                        overlay = cv2.addWeighted(
                            cv2.cvtColor(cca_result['original_mask'], cv2.COLOR_GRAY2RGB), 
                            0.5, 
                            colored_labels, 
                            0.5, 
                            0
                        )
                        axes[2].imshow(overlay)
                    else:
                        axes[2].imshow(cca_result['original_mask'], cmap='gray')
                    axes[2].set_title("Overlay")
                    axes[2].axis('off')
                    
                    plt.suptitle(f"Sample {i+1}: {sample['filename']}\n{tumor_type.upper()}", 
                               fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    
                    # Save figure
                    safe_filename = sample['filename'].replace('.', '_')
                    plt.savefig(vis_dir / f"{tumor_type}_sample_{i+1}_{safe_filename}.png", 
                              dpi=150, bbox_inches='tight')
                    plt.close()
        
        print(f"\nSample visualizations saved to {vis_dir}/")


# Main function with debugging
def main():
    """
    Main function to run CCA analysis with debugging
    """
    print("=" * 70)
    print("CCA ANALYSIS FOR ALL TUMOR TYPES")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = CCAAnalyzer(base_path="segmented_output")
    
    # First, let's debug the directory structure
    print("\n[DEBUG] Checking directory structure...")
    analyzer.verify_directory_structure()
    
    # Find mask files
    print("\n[DEBUG] Searching for mask files...")
    mask_files = analyzer.find_mask_files(verbose=True)
    
    if not mask_files:
        print("\nERROR: No mask files found!")
        print("\nPlease check:")
        print("1. The 'segmented_output' directory exists")
        print("2. It contains 'Training' folder or tumor type folders directly")
        print("3. Each tumor folder contains 'masks' folder or mask files directly")
        return
    
    # Analyze a small sample from each type first (for testing)
    print("\n[TEST] Analyzing small sample from each tumor type...")
    analyzer.analyze_all_masks(
        min_size=50,
        max_samples_per_type=10,  # Only 10 per type for testing
        verbose=True
    )
    
    if analyzer.results:
        # Export test results
        analyzer.export_results(output_dir="cca_test_results")
        
        # Ask user if they want to process all files
        response = input("\nDo you want to process ALL files? (y/n): ")
        if response.lower() == 'y':
            print("\nProcessing ALL files...")
            analyzer.analyze_all_masks(min_size=50, max_samples_per_type=None, verbose=True)
            analyzer.export_results(output_dir="cca_full_results")
    else:
        print("\nERROR: No results generated!")
        print("\nPossible issues:")
        print("1. Mask files might be corrupted or in wrong format")
        print("2. File permissions might be restricting access")
        print("3. The images might not be binary masks")


# Quick test function
def quick_test():
    """Quick test to verify the script works"""
    print("Running quick test...")
    
    analyzer = CCAAnalyzer(base_path="segmented_output")
    
    # Debug directory structure
    analyzer.verify_directory_structure()
    
    # Find files
    mask_files = analyzer.find_mask_files(verbose=True)
    
    if mask_files:
        print(f"\nFound {len(mask_files)} total mask files")
        
        # Process just 2 files from each type
        for tumor_type in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            type_files = [m for m in mask_files if m['type'] == tumor_type]
            print(f"\nProcessing 2 samples from {tumor_type}:")
            
            for i, mask_info in enumerate(type_files[:2]):
                print(f"  {i+1}. {mask_info['filename']}")
                result = analyzer.apply_cca(mask_info['path'], min_size=50)
                if result:
                    print(f"     Components: {result['num_components']}, "
                          f"Total area: {sum([c['area'] for c in result['components']])} px²")
                else:
                    print("     Failed to process")
    else:
        print("No mask files found!")


if __name__ == "__main__":
    # Run the main analysis
    main()
    
    # Or run quick test
    # quick_test()