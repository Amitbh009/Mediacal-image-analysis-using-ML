# Medical Image Analysis using Machine Learning

## Overview

This repository presents a **Medical Image Analysis pipeline using Machine Learning techniques** for analyzing brain MRI images. The project demonstrates how image preprocessing, enhancement, feature extraction, dimensionality reduction, segmentation, and classification can be combined to assist in identifying patterns in medical images.

The implementation explores classical machine learning methods along with image processing techniques to build a **complete workflow for medical image analysis and classification**.

---

## Project Pipeline

The system follows a structured workflow commonly used in medical image processing:

1. **Image Preprocessing**

   * Image normalization
   * Noise reduction
   * Image quality enhancement

2. **Image Enhancement**

   * Wavelet-based enhancement for improving important structures.

3. **Segmentation**

   * K-Means clustering used to segment important regions in MRI images.

4. **Feature Extraction**

   * Extract meaningful features from segmented images.

5. **Dimensionality Reduction**

   * Principal Component Analysis (PCA) used to reduce feature dimensions.

6. **Classification**

   * Support Vector Machine (SVM) used to classify images.

7. **Evaluation**

   * Performance analysis of the trained model.

---

## Repository Structure

```
Medical-image-analysis-using-ML
│
├── notebook
│   └── medical_image_analysis_using_ml.ipynb
│
├── preprocessing
│   ├── preprocess.py
│   ├── check_preprocess.py
│
├── enhancement
│   └── wavelet_enhancement.py
│
├── segmentation
│   ├── kmeans_segment.py
│   └── segment_check.py
│
├── feature_extraction
│   └── feature_extraction.py
│
├── dimensionality_reduction
│   └── principal_component_analysis.py
│
├── classification
│   └── svm.py
│
├── analysis
│   ├── snr.py
│   └── cca.py
│
├── dataset
│   └── dataset
│
└── README.md
```

---

## Description of Key Files

### Preprocessing

* **preprocess.py**
  Performs image preprocessing steps such as resizing, normalization, and preparing the dataset.

* **check_preprocess.py**
  Used to verify the correctness of preprocessing operations.

---

### Image Enhancement

* **wavelet_enhancement.py**
  Applies wavelet-based techniques to enhance important image features.

---

### Segmentation

* **kmeans_segment.py**
  Implements K-Means clustering to segment the MRI images.

* **segment_check.py**
  Used to validate segmentation output.

---

### Feature Extraction

* **feature_extraction.py**
  Extracts relevant features from the processed images for machine learning models.

---

### Dimensionality Reduction

* **principal_component_analysis.py**
  Applies PCA to reduce feature dimensionality while preserving useful information.

---

### Classification

* **svm.py**
  Implements Support Vector Machine for classification of medical images.

---

### Additional Analysis

* **snr.py**
  Calculates Signal-to-Noise Ratio to evaluate image quality.

* **cca.py**
  Performs Canonical Correlation Analysis for feature relationship analysis.

---

## Technologies Used

* Python
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* Google Colab / Jupyter Notebook

---

## Applications

This project demonstrates the role of machine learning in:

* Medical image analysis
* Brain MRI interpretation
* Computer-aided diagnosis systems
* Medical research and healthcare AI

---


## Disclaimer

This repository is created for **academic and research purposes only** and should not be used for real medical diagnosis.
