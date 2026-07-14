import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc, 
    roc_auc_score, precision_recall_curve, average_precision_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

PCA_DATA_PATH = "/content/pca_analysis/transformed_features.csv"
SVM_OUTPUT_ROOT = "/content/svm_binary_tumor_classification"
os.makedirs(SVM_OUTPUT_ROOT, exist_ok=True)

# -------- BINARY LABEL CREATION --------
def create_binary_labels(df):
    """
    Convert 16-class labels to binary tumor/non-tumor labels
    
    Based on your class names:
    - 'no' likely means 'no tumor'
    - 'gl' (glioma), 'me' (meningioma), 'pi' (pituitary) likely mean tumor types
    """
    print("=" * 80)
    print("CREATING BINARY LABELS (TUMOR vs NON-TUMOR)")
    print("=" * 80)
    if 'label' in df.columns:
        labels = df['label'].values
    elif 'class' in df.columns:
        labels = df['class'].values
    elif 'image_name' in df.columns:
        labels = df['image_name'].values
    else:
        print("❌ No label column found")
        return None
    
    print(f"Original labels: {np.unique(labels[:5])}...") 
    
    tumor_classes = ['gl', 'me', 'pi'] 
    non_tumor_classes = ['no']
    
    binary_labels = []
    tumor_count = 0
    non_tumor_count = 0
    
    for label in labels:
        label_str = str(label).lower()
        
        is_tumor = any(tumor_class in label_str for tumor_class in tumor_classes)
        is_non_tumor = any(non_tumor_class in label_str for non_tumor_class in non_tumor_classes)
        
        if is_tumor:
            binary_labels.append('tumor')
            tumor_count += 1
        elif is_non_tumor:
            binary_labels.append('non_tumor')
            non_tumor_count += 1
        else:
            print(f"⚠️ Unknown label type: {label}")
            binary_labels.append('unknown')
    
    binary_labels = np.array(binary_labels)
    
    print(f"\n📊 Binary Label Distribution:")
    print(f"  Tumor samples: {tumor_count} ({tumor_count/len(labels)*100:.1f}%)")
    print(f"  Non-tumor samples: {non_tumor_count} ({non_tumor_count/len(labels)*100:.1f}%)")
    
    if 'unknown' in binary_labels:
        known_mask = binary_labels != 'unknown'
        binary_labels = binary_labels[known_mask]
        print(f"  Removed {np.sum(~known_mask)} unknown samples")
    
    return binary_labels

def load_and_prepare_binary_data(pca_data_path, n_components=20):
    """
    Load PCA data and create binary tumor/non-tumor classification
    """
    print("=" * 80)
    print("LOADING DATA FOR BINARY TUMOR CLASSIFICATION")
    print("=" * 80)
    
    if not os.path.exists(pca_data_path):
        print(f"❌ PCA data not found: {pca_data_path}")
        return None, None, None, None
    
    df = pd.read_csv(pca_data_path)
    
    pc_columns = [col for col in df.columns if col.startswith('PC_')]
    if n_components is not None:
        pc_columns = pc_columns[:n_components]
    
    X = df[pc_columns].values
    feature_names = pc_columns
    
    print(f"PCA data shape: {X.shape}")
    print(f"Using {len(feature_names)} PCA components")
    
    y_binary = create_binary_labels(df)
    
    if y_binary is None:
        print("❌ Failed to create binary labels")
        return None, None, None, None
    
    if len(y_binary) != len(X):
        print(f"⚠️ Label count ({len(y_binary)}) doesn't match feature count ({len(X)})")
        print(f"Using first {len(y_binary)} samples from features")
        X = X[:len(y_binary), :]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binary)
    class_names = le.classes_
    
    print(f"\n📊 Final Dataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(class_names)}")
    for i, class_name in enumerate(class_names):
        count = np.sum(y_encoded == i)
        percentage = (count / len(y_encoded)) * 100
        print(f"    - {class_name}: {count} samples ({percentage:.1f}%)")
    
    return X, y_encoded, class_names, feature_names

def train_binary_svm(X_train, y_train, X_val, y_val, fast_mode=True):
    """Train SVM for binary classification with optimized settings"""
    print("\n" + "=" * 80)
    print("TRAINING BINARY SVM (TUMOR vs NON-TUMOR)")
    print("=" * 80)
    
    if fast_mode:
        print("Using fast training mode...")
        
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf']
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42, class_weight='balanced'),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
    else:
        print("Training with default parameters...")
        best_model = SVC(
            C=1.0, 
            kernel='rbf', 
            gamma='scale', 
            probability=True, 
            random_state=42,
            class_weight='balanced'
        )
        best_model.fit(X_train, y_train)
    
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='binary')
    val_recall = recall_score(y_val, y_val_pred, average='binary')
    val_f1 = f1_score(y_val, y_val_pred, average='binary')
    
    print(f"\n📊 Validation Set Performance:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    
    return best_model

def evaluate_binary_model(model, X_test, y_test, class_names, output_dir):
    """Enhanced evaluation for binary classification"""
    print("\n" + "=" * 80)
    print("BINARY MODEL EVALUATION ON TEST SET")
    print("=" * 80)
    
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"               Predicted")
    print(f"              {class_names[0]:<10} {class_names[1]:<10}")
    print(f"Actual {class_names[0]:<5} {cm[0,0]:<10} {cm[0,1]:<10}")
    print(f"       {class_names[1]:<5} {cm[1,0]:<10} {cm[1,1]:<10}")
    
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall 
    ppv = precision 
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n📈 Additional Metrics:")
    print(f"  Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"  Specificity (True Negative Rate): {specificity:.4f}")
    print(f"  Positive Predictive Value (Precision): {ppv:.4f}")
    print(f"  Negative Predictive Value: {npv:.4f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "  False Positive Rate: 0.0000")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = os.path.join(eval_dir, "binary_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n💾 Metrics saved to: {metrics_csv}")
    
    create_binary_visualizations(cm, y_test, y_pred_proba, class_names, eval_dir, metrics)
    
    return metrics, cm

def create_binary_visualizations(cm, y_test, y_pred_proba, class_names, output_dir, metrics):
    """Create visualizations for binary classification"""
    print("\n📈 Creating binary classification visualizations...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix\nAccuracy: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
            label=f'Optimal threshold: {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, 
            label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
    plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal threshold ({optimal_threshold:.2f})')
    plt.xlabel('Predicted Probability of Tumor')
    plt.ylabel('Count')
    plt.title('Distribution of Tumor Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    correct_mask = (y_pred_proba >= 0.5) == y_test
    plt.hist(y_pred_proba[correct_mask], bins=20, alpha=0.7, 
            color='green', label='Correct', edgecolor='black')
    plt.hist(y_pred_proba[~correct_mask], bins=20, alpha=0.7, 
            color='red', label='Incorrect', edgecolor='black')
    plt.xlabel('Predicted Probability of Tumor')
    plt.ylabel('Count')
    plt.title('Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_distribution.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1_score'], metrics['specificity']]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
    bars = plt.bar(metric_names, metric_values, color=colors, edgecolor='black')
    
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.title('Binary Classification Performance Metrics')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_metrics.png"), dpi=150)
    plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")

def run_binary_svm_pipeline(pca_data_path, n_components=20, fast_mode=True):
    """Complete pipeline for binary tumor/non-tumor classification"""
    
    print("=" * 80)
    print("BINARY TUMOR/NON-TUMOR SVM CLASSIFICATION PIPELINE")
    print("=" * 80)
    
    X, y, class_names, feature_names = load_and_prepare_binary_data(
        pca_data_path, n_components
    )
    
    if X is None:
        print("❌ Failed to load data. Exiting.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train 
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {X_train_final.shape[0]} samples ({X_train_final.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    svm_model = train_binary_svm(X_train_final, y_train_final, X_val, y_val, fast_mode)
    
    metrics, cm = evaluate_binary_model(svm_model, X_test, y_test, class_names, SVM_OUTPUT_ROOT)
    
    model_path = os.path.join(SVM_OUTPUT_ROOT, "binary_svm_model.pkl")
    joblib.dump(svm_model, model_path)
    
    model_info = {
        'model_type': 'SVM',
        'task': 'binary_tumor_classification',
        'classes': class_names.tolist(),
        'class_mapping': {'0': class_names[0], '1': class_names[1]},
        'n_features': len(feature_names),
        'n_components': n_components,
        'performance': metrics
    }
    
    import json
    info_path = os.path.join(SVM_OUTPUT_ROOT, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("BINARY SVM PIPELINE COMPLETE!")
    print("=" * 80)
    
    summary = {
        'model_path': model_path,
        'accuracy': metrics['accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'f1_score': metrics['f1_score'],
        'n_components': n_components,
        'output_dir': SVM_OUTPUT_ROOT
    }
    
    print("\n📊 FINAL SUMMARY:")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n🏥 CLINICAL INTERPRETATION:")
    print(f"  • Model correctly identifies {metrics['sensitivity']*100:.1f}% of tumors")
    print(f"  • Model correctly identifies {metrics['specificity']*100:.1f}% of non-tumors")
    print(f"  • When model predicts tumor, it's correct {metrics['precision']*100:.1f}% of the time")
    
    return svm_model, summary

def run_quick_binary_classification():
    """Quick function to run binary classification"""
    print("Starting binary tumor/non-tumor classification...")
    
    model, summary = run_binary_svm_pipeline(
        pca_data_path=PCA_DATA_PATH,
        n_components=20,  
        fast_mode=True    
    )
    
    if model:
        print(f"\n✅ Binary tumor classification complete!")
        print(f"📁 Output directory: {SVM_OUTPUT_ROOT}")
        print(f"🎯 Model accuracy: {summary['accuracy']:.4f}")
    
    return model, summary

if __name__ == "__main__":
    model, summary = run_quick_binary_classification()