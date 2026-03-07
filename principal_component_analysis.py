import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import joblib
from scipy import stats

FEATURES_PATH = "/content/features_complete/features_ALL_complete.csv"
PCA_OUTPUT_ROOT = "/content/pca_analysis"
os.makedirs(PCA_OUTPUT_ROOT, exist_ok=True)

def load_and_prepare_features(features_path):
    """Load features and prepare for PCA"""
    print("=" * 80)
    print("LOADING FEATURES FOR PCA ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv(features_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1] - 5}") 
    
    metadata_cols = ['image_name', 'image_path', 'mask_shape', 'original_found', 'enhanced_found']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"\n📊 Feature types:")
    print(f"  - Metadata columns: {len(metadata_cols)}")
    print(f"  - Feature columns: {len(feature_cols)}")
    
    X = df[feature_cols].values
    feature_names = feature_cols
    
    missing = pd.DataFrame(X).isnull().sum().sum()
    if missing > 0:
        print(f"\n⚠️ Found {missing} missing values. Filling with median...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
    
    if np.any(np.isinf(X)):
        print("⚠️ Found infinite values. Replacing with large finite values...")
        X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
    
    variances = np.var(X, axis=0)
    constant_mask = variances == 0
    if np.any(constant_mask):
        constant_features = np.array(feature_names)[constant_mask]
        print(f"\n⚠️ Removing {len(constant_features)} constant features: {list(constant_features)}")
        X = X[:, ~constant_mask]
        feature_names = np.array(feature_names)[~constant_mask].tolist()
    
    print(f"\n✅ Final feature matrix shape: {X.shape}")
    print(f"✅ Feature names: {len(feature_names)}")
    
    return X, feature_names, df, metadata_cols

def scale_features(X, feature_names):
    """Standardize features for PCA"""
    print("\n" + "=" * 80)
    print("FEATURE STANDARDIZATION")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Mean after scaling: {X_scaled.mean():.6f} (should be ~0)")
    print(f"Std after scaling: {X_scaled.std():.6f} (should be ~1)")
    
    scaler_path = os.path.join(PCA_OUTPUT_ROOT, "standard_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")
    
    return X_scaled, scaler

def perform_pca_analysis(X_scaled, feature_names, n_components=None):
    """Perform PCA analysis"""
    print("\n" + "=" * 80)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 80)
    
    if n_components is None:
        n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"PCA dimensions: {X_pca.shape[1]}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    print("\n📈 Variance explained by components:")
    for i, (exp_var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
        if i < 10 or (i + 1) % 10 == 0 or cum_var >= 0.99:
            print(f"  PC{i+1:3d}: {exp_var:.4f} ({cum_var:.4f} cumulative)")
    
    print("\n🎯 Components needed for variance thresholds:")
    for threshold in thresholds:
        n_components_needed = np.argmax(cumulative_variance >= threshold) + 1
        print(f"  {threshold*100:.0f}% variance: {n_components_needed} components")
    
    pca_path = os.path.join(PCA_OUTPUT_ROOT, "pca_model.pkl")
    joblib.dump(pca, pca_path)
    print(f"\n💾 PCA model saved to: {pca_path}")
    
    return X_pca, pca, cumulative_variance

def visualize_pca_results(X_pca, pca, X_scaled, feature_names, cumulative_variance):
    """Create comprehensive PCA visualizations"""
    print("\n" + "=" * 80)
    print("PCA VISUALIZATIONS")
    print("=" * 80)
    
    viz_dir = os.path.join(PCA_OUTPUT_ROOT, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    n_components = X_pca.shape[1]
    
    print("1. Creating Scree Plot...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                  pca.explained_variance_ratio_[:20], 
                  alpha=0.7, color='steelblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot (First 20 PCs)')
    plt.xticks(range(1, 21))
    plt.grid(True, alpha=0.3)
    
    for bar in bars[:10]:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
            'b-o', linewidth=2, markersize=4)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)
    
    thresholds = [0.80, 0.85, 0.90, 0.95]
    colors = ['r--', 'g--', 'b--', 'm--']
    for threshold, color in zip(thresholds, colors):
        n_comp = np.argmax(cumulative_variance >= threshold) + 1
        plt.axvline(x=n_comp, linestyle='--', color=color[0], alpha=0.7, 
                   label=f'{threshold*100:.0f}% ({n_comp} PCs)')
        plt.axhline(y=threshold, linestyle='--', color=color[0], alpha=0.3)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "scree_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("2. Creating Biplot...")
    if n_components >= 2:
        plt.figure(figsize=(14, 10))
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            alpha=0.6, s=20, c='blue', edgecolors='k', linewidth=0.5)
        
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        feature_importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
        top_n = min(20, len(feature_names))
        top_indices = np.argsort(feature_importance)[-top_n:]
        
        scale_factor = 3.0 / loadings[top_indices].max()
        
        for idx in top_indices:
            plt.arrow(0, 0, 
                     loadings[idx, 0] * scale_factor, 
                     loadings[idx, 1] * scale_factor,
                     color='red', alpha=0.7, width=0.005, 
                     head_width=0.03, head_length=0.03)
            
            plt.text(loadings[idx, 0] * scale_factor * 1.15,
                    loadings[idx, 1] * scale_factor * 1.15,
                    feature_names[idx], fontsize=9, color='darkred',
                    ha='center', va='center')
        
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
        plt.title('PCA Biplot: Samples and Feature Loadings')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, alpha=0.3, linestyle='--')
        plt.gca().add_artist(circle)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "pca_biplot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print("3. Creating Loadings Heatmap...")
    n_top_components = min(10, n_components)
    
    n_top_features = min(15, len(feature_names))
    
    loadings_df = pd.DataFrame(
        pca.components_[:n_top_components, :].T,
        columns=[f'PC{i+1}' for i in range(n_top_components)],
        index=feature_names
    )
    
    feature_scores = np.abs(loadings_df.values).sum(axis=1)
    top_feature_indices = np.argsort(feature_scores)[-n_top_features:]
    top_loadings_df = loadings_df.iloc[top_feature_indices]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_loadings_df, 
                cmap='RdBu_r', 
                center=0,
                annot=True, 
                fmt='.2f',
                cbar_kws={'label': 'Loading Coefficient'},
                square=False,
                linewidths=0.5)
    
    plt.title(f'PCA Loadings Heatmap (Top {n_top_features} Features)')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "loadings_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("4. Creating 3D PCA Plot...")
    if n_components >= 3:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                           alpha=0.6, s=20, c=X_pca[:, 2], cmap='viridis',
                           edgecolors='k', linewidth=0.3)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
        ax.set_title('3D PCA Projection')
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, label='PC3 Value')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "pca_3d.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print("5. Creating Feature Importance Chart...")
    feature_importance = np.abs(pca.components_).sum(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    
    plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
    plt.xlabel('Cumulative Absolute Loading')
    plt.title('Top 20 Features by PCA Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(importance * 1.01, i, f'{importance:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {viz_dir}")
    return viz_dir

def apply_pca_reduction(X_scaled, pca, variance_threshold=0.95):
    """Apply PCA to reduce dimensions"""
    print("\n" + "=" * 80)
    print("APPLYING PCA FOR DIMENSIONALITY REDUCTION")
    print("=" * 80)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Variance threshold: {variance_threshold*100:.0f}%")
    print(f"Components needed: {n_components}")
    print(f"Variance explained: {cumulative_variance[n_components-1]:.4f}")
    
    X_reduced = pca.transform(X_scaled)[:, :n_components]
    
    print(f"\n📊 Dimensionality Reduction:")
    print(f"  Original dimensions: {X_scaled.shape[1]}")
    print(f"  Reduced dimensions: {X_reduced.shape[1]}")
    print(f"  Compression ratio: {X_scaled.shape[1]/X_reduced.shape[1]:.1f}:1")
    print(f"  Data reduction: {(1 - X_reduced.shape[1]/X_scaled.shape[1])*100:.1f}%")
    
    return X_reduced, n_components

def save_pca_results(X_reduced, original_df, metadata_cols, n_components, feature_names, pca):
    """Save PCA-transformed data"""
    print("\n" + "=" * 80)
    print("SAVING PCA RESULTS")
    print("=" * 80)
    
    pca_columns = [f'PC_{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_reduced, columns=pca_columns)
    
    for col in metadata_cols:
        if col in original_df.columns:
            pca_df[col] = original_df[col].values
    
    pca_csv_path = os.path.join(PCA_OUTPUT_ROOT, "pca_transformed_data.csv")
    pca_df.to_csv(pca_csv_path, index=False)
    
    thresholds = [0.80, 0.85, 0.90, 0.95]
    for threshold in thresholds:
        n_comp = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1
        X_threshold = pca.transform(original_df.drop(columns=metadata_cols).values)[:, :n_comp]
        threshold_df = pd.DataFrame(X_threshold, columns=[f'PC_{i+1}' for i in range(n_comp)])
        for col in metadata_cols:
            if col in original_df.columns:
                threshold_df[col] = original_df[col].values
        
        threshold_path = os.path.join(PCA_OUTPUT_ROOT, f"pca_{int(threshold*100)}percent.csv")
        threshold_df.to_csv(threshold_path, index=False)
        print(f"  ✅ Saved {threshold*100:.0f}% variance data ({n_comp} PCs): {threshold_path}")
    
    loadings_df = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=pca_columns,
        index=feature_names[:len(pca.components_.T)]
    )
    loadings_csv = os.path.join(PCA_OUTPUT_ROOT, "pca_loadings.csv")
    loadings_df.to_csv(loadings_csv)
    
    variance_df = pd.DataFrame({
        'PC': range(1, len(pca.explained_variance_ratio_) + 1),
        'Explained_Variance': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    variance_csv = os.path.join(PCA_OUTPUT_ROOT, "pca_variance.csv")
    variance_df.to_csv(variance_csv, index=False)
    
    print(f"\n💾 PCA Results saved:")
    print(f"  - Transformed data: {pca_csv_path}")
    print(f"  - PCA loadings: {loadings_csv}")
    print(f"  - Variance explained: {variance_csv}")
    print(f"  - Different variance thresholds saved separately")
    
    return pca_df, loadings_df, variance_df

def generate_pca_report(pca, feature_names, n_components, variance_df):
    """Generate comprehensive PCA report"""
    print("\n" + "=" * 80)
    print("PCA ANALYSIS REPORT")
    print("=" * 80)
    
    report_path = os.path.join(PCA_OUTPUT_ROOT, "pca_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  - Total samples: {pca.n_samples_}\n")
        f.write(f"  - Original features: {len(feature_names)}\n")
        f.write(f"  - PCA components extracted: {n_components}\n")
        f.write(f"  - Total variance explained: {pca.explained_variance_ratio_.sum():.4f}\n\n")
        
        f.write("Top 10 Principal Components:\n")
        for i in range(min(10, n_components)):
            f.write(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
                   f"({np.cumsum(pca.explained_variance_ratio_)[i]:.4f} cumulative)\n")
        
        f.write("\nFeature Importance (Top 10 features for each of first 3 PCs):\n")
        for pc_idx in range(min(3, n_components)):
            f.write(f"\n  PC{pc_idx+1} (Variance: {pca.explained_variance_ratio_[pc_idx]:.4f}):\n")
            pc_loadings = pca.components_[pc_idx]
            top_indices = np.argsort(np.abs(pc_loadings))[-10:][::-1]
            
            for rank, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                f.write(f"    {rank+1:2d}. {feature_name:30s}: {pc_loadings[idx]:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n")
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        f.write("\n1. For visualization (2D/3D plots):\n")
        f.write("   - Use PC1, PC2, PC3 (explains {:.1f}% variance)\n".format(
            cumulative_variance[2] * 100))
        
        f.write("\n2. For dimensionality reduction:\n")
        for threshold in [0.80, 0.85, 0.90, 0.95]:
            n_comp = np.argmax(cumulative_variance >= threshold) + 1
            f.write(f"   - {threshold*100:.0f}% variance: {n_comp} components "
                   f"(reduction: {(1 - n_comp/len(feature_names))*100:.1f}%)\n")
        
        f.write("\n3. For feature selection:\n")
        feature_importance = np.abs(pca.components_).sum(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        f.write("   Top 10 most influential features:\n")
        for i, idx in enumerate(top_features_idx):
            f.write(f"   {i+1:2d}. {feature_names[idx]}\n")
    
    print(f"📄 PCA report saved to: {report_path}")
    
    with open(report_path, 'r') as f:
        lines = f.readlines()
        for line in lines[:50]: 
            print(line.rstrip())
    
    return report_path

def run_complete_pca_pipeline(features_path, variance_threshold=0.95):
    """Complete PCA pipeline from features to reduced dimensions"""
    
    print("=" * 80)
    print("COMPLETE PCA PIPELINE FOR FEATURE SELECTION")
    print("=" * 80)
    
    X, feature_names, original_df, metadata_cols = load_and_prepare_features(features_path)
    
    X_scaled, scaler = scale_features(X, feature_names)
    
    X_pca, pca, cumulative_variance = perform_pca_analysis(X_scaled, feature_names)
    
    viz_dir = visualize_pca_results(X_pca, pca, X_scaled, feature_names, cumulative_variance)
    
    X_reduced, n_components = apply_pca_reduction(X_scaled, pca, variance_threshold)
    
    pca_df, loadings_df, variance_df = save_pca_results(
        X_reduced, original_df, metadata_cols, n_components, feature_names, pca
    )
    
    report_path = generate_pca_report(pca, feature_names, n_components, variance_df)
    
    print("\n" + "=" * 80)
    print("PCA PIPELINE COMPLETE!")
    print("=" * 80)
    
    summary = {
        'original_features': X.shape[1],
        'reduced_features': n_components,
        'variance_explained': cumulative_variance[n_components-1],
        'compression_ratio': X.shape[1] / n_components,
        'data_reduction': f"{(1 - n_components/X.shape[1])*100:.1f}%",
        'output_dir': PCA_OUTPUT_ROOT,
        'visualizations': viz_dir,
        'report': report_path
    }
    
    print("\n📊 FINAL SUMMARY:")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return pca_df, pca, summary

def quick_pca_analysis():
    """Quick PCA analysis with default settings"""
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Features file not found: {FEATURES_PATH}")
        return None
    
    print("Running quick PCA analysis...")
    return run_complete_pca_pipeline(FEATURES_PATH, variance_threshold=0.95)

def analyze_existing_pca():
    """Analyze existing PCA results"""
    pca_csv = os.path.join(PCA_OUTPUT_ROOT, "pca_transformed_data.csv")
    loadings_csv = os.path.join(PCA_OUTPUT_ROOT, "pca_loadings.csv")
    variance_csv = os.path.join(PCA_OUTPUT_ROOT, "pca_variance.csv")
    
    if not os.path.exists(pca_csv):
        print("❌ No existing PCA results found. Run the pipeline first.")
        return None
    
    print("Loading existing PCA results...")
    
    pca_df = pd.read_csv(pca_csv)
    loadings_df = pd.read_csv(loadings_csv, index_col=0)
    variance_df = pd.read_csv(variance_csv)
    
    print(f"📊 PCA Transformed Data: {pca_df.shape}")
    print(f"📊 PCA Loadings: {loadings_df.shape}")
    print(f"📊 Variance Explained: {len(variance_df)} components")
    
    # Plot variance
    plt.figure(figsize=(10, 6))
    plt.plot(variance_df['PC'], variance_df['Cumulative_Variance'], 'b-o', linewidth=2)
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('PCA Cumulative Variance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pca_df, loadings_df, variance_df

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING PCA FEATURE SELECTION")
    print("=" * 80)
    
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Features file not found: {FEATURES_PATH}")
        print("Please ensure feature extraction is complete.")
    else:
        pca_results, pca_model, summary = run_complete_pca_pipeline(
            FEATURES_PATH, 
            variance_threshold=0.95
        )
        
        print("\n✅ PCA analysis complete!")
        print(f"📁 Output directory: {PCA_OUTPUT_ROOT}")
        print(f"📊 Original features: {summary['original_features']}")
        print(f"📊 Reduced features: {summary['reduced_features']}")
        print(f"📊 Variance retained: {summary['variance_explained']*100:.1f}%")
