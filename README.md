# Handwritten Digit Image Classification Project

An end-to-end, reproducible, leakage-safe machine learning classification project built in strict adherence to the **Programming in Python | Final-Term Project Requirements** ruleset.

---

## 1. Project Overview & Workflow

This project implements a multi-class image classification workflow to recognize handwritten digits (0 through 9) from normalized 8x8 pixel intensity matrices.

### Core Workflow Stages:
- **Data Provenance & Audit**: Automated loading and integrity checks (ranges, missing values, duplicates, and class balance).
- **Leakage-Safe Splitting**: 80/20 stratified train/test split with quarantined test data.
- **Exploratory Data Analysis**: Visualizing class balance, sample digits, mean class heatmaps, and intensity distributions.
- **Preprocessing Pipeline**: `MinMaxScaler` feature scaling encapsulated within scikit-learn `Pipeline` objects.
- **Model Development & CV Tuning**: 5-Fold Stratified Cross-Validation tuning across Logistic Regression, Random Forest, and Support Vector Machine against a Majority Baseline.
- **Final Evaluation & Error Analysis**: Single evaluation pass on untouched test data with confusion matrices, class-wise metrics, and misclassified image diagnostics.

---

## 2. Quick Start & Setup

### Prerequisites
- Python 3.10+ (Anaconda or standard Python virtual environment)

### Installation
1. Open a terminal / PowerShell in the project directory (`d:\Python_Project`).
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. How to Run

### Option A: Command-Line Execution (Recommended)
Run the automated end-to-end pipeline:
```bash
python main.py
```
This will:
1. Load raw data and save a persistent copy to `data/raw_digits.csv`.
2. Perform leakage-safe stratified splitting (`random_state = 42`).
3. Generate and save exploratory figures (`outputs/figures/01` to `04`).
4. Train models and tune hyperparameters via 5-Fold Stratified Cross-Validation.
5. Evaluate performance on the untouched test split.
6. Generate diagnostic figures (`outputs/figures/05` to `08`) and export `outputs/test_evaluation_summary.csv`.

### Option B: Interactive Jupyter Notebook
Open and run the notebook:
```bash
jupyter notebook notebook.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells sequentially.

---

## 4. Project Directory Structure

```text
d:\Python_Project\
├── .gitignore                # Git ignore rules for Python/ML projects
├── config.py                 # Centralized seeds, paths, and constants
├── main.py                   # Automated end-to-end pipeline script
├── notebook.ipynb            # Interactive step-by-step Jupyter Notebook
├── requirements.txt          # Package dependencies
├── data_dictionary.md        # Feature & target data dictionary
├── README.md                 # Project documentation and run guide
├── PROJECT_REPORT.md         # Full 12-section project report (PDF compliant)
├── data\                     # Raw data directory
│   └── raw_digits.csv        # Persisted raw dataset copy
├── outputs\                  # Generated metrics and visual artifacts
│   ├── figures\              # High-resolution (300 DPI) plots
│   │   ├── 01_class_distribution.png
│   │   ├── 02_sample_digits.png
│   │   ├── 03_mean_digit_heatmaps.png
│   │   ├── 04_pixel_intensity_distribution.png
│   │   ├── 05_confusion_matrices.png
│   │   ├── 06_model_comparison.png
│   │   ├── 07_misclassified_examples.png
│   │   └── 08_classwise_f1_scores.png
│   └── test_evaluation_summary.csv
└── src\                      # Modular Python source package
    ├── __init__.py
    ├── data_loader.py        # Data ingestion, audit, and stratified split
    ├── eda.py                # Exploratory data analysis plots
    ├── models.py             # Preprocessing pipelines & CV tuning
    ├── evaluate.py           # Test set evaluation & error extraction
    └── visualize.py          # Diagnostic plots & error visualization
```

---

## 5. Summary of Final Results

### Benchmark Metrics (Untouched 20% Test Split)

| Model Family | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (OVR) | Fit Time (s) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority)** | 10.00% | 0.0100 | 0.1000 | 0.0182 | N/A | 0.01 s | 0.002 ms |
| **Logistic Regression** | 96.11% | 0.9620 | 0.9610 | 0.9612 | 0.9982 | 0.85 s | 0.004 ms |
| **Random Forest** | 97.50% | 0.9758 | 0.9749 | 0.9751 | 0.9995 | 1.12 s | 0.038 ms |
| **Support Vector Machine** | **98.61%** | **0.9868** | **0.9860** | **0.9863** | **0.9998** | **0.32 s** | **0.018 ms** |

### Key Takeaways:
- **Top Performer**: Support Vector Machine (SVC with RBF kernel and MinMax scaling) achieved **98.61% test accuracy** with only 5 misclassified test samples out of 360.
- **Fast Execution**: Entire pipeline trains and evaluates in under 3 seconds total.
- **Leakage Prevention**: Strictly zero test data influence on scaling or parameter selection.

---

## 6. Generated Visual Artifacts Index

| Figure File | Description |
| :--- | :--- |
| `01_class_distribution.png` | Stratified train (80%) and test (20%) class balance chart. |
| `02_sample_digits.png` | Representative 8x8 handwritten digit samples for classes 0–9. |
| `03_mean_digit_heatmaps.png` | Average pixel intensity profile heatmaps per digit class. |
| `04_pixel_intensity_distribution.png` | Histogram showing raw pixel intensity distribution [0–16]. |
| `05_confusion_matrices.png` | Test confusion matrix heatmaps for candidate models. |
| `06_model_comparison.png` | Side-by-side comparison of Accuracy, Macro F1, and training time. |
| `07_misclassified_examples.png` | Diagnostic gallery of actual misclassified test images with True vs Predicted labels. |
| `08_classwise_f1_scores.png` | Per-class F1-score breakdown across all 10 digit classes. |

---

## 7. Relevant Documents
- [Full Project Report (PROJECT_REPORT.md)](file:///d:/Python_Project/PROJECT_REPORT.md)
- [Data Dictionary (data_dictionary.md)](file:///d:/Python_Project/data_dictionary.md)
