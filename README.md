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
- **Final Evaluation & Error Analysis**: Single evaluation pass on untouched test data with confusion matrices, class-wise precision/recall/F1 metrics, and misclassified image diagnostics.

---

## 2. Quick Start & Setup

### Environment Specifications
- **Python**: `3.13.5` (or Python 3.10+)
- **Packages**:
  - `scikit-learn`: `1.6.1`
  - `numpy`: `2.3.1`
  - `pandas`: `2.2.3`
  - `matplotlib`: `3.10.0`
  - `seaborn`: `0.13.2`
  - `joblib`: `1.5.2`

### Installation
1. Open a terminal / PowerShell in the project root directory.
2. Install dependencies:
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
6. Print overall model benchmarks and class-wise precision/recall/F1 metrics.
7. Generate diagnostic figures (`outputs/figures/05` to `08`) and export CSV summaries to `outputs/`.

### Option B: Interactive Jupyter Notebook
Open and run the notebook:
```bash
jupyter notebook notebook.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells sequentially.

---

## 4. Project Directory Structure

```text
project_root/
├── .gitignore                # Git ignore rules for Python/ML projects
├── config.py                 # Centralized seeds, relative paths, and constants
├── main.py                   # Automated end-to-end pipeline script
├── notebook.ipynb            # Interactive step-by-step Jupyter Notebook
├── requirements.txt          # Package dependencies
├── data_dictionary.md        # Feature & target data dictionary
├── README.md                 # Project documentation and run guide
├── PROJECT_REPORT.md         # Full 12-section project report (PDF compliant)
├── data/                     # Raw data directory
│   └── raw_digits.csv        # Persisted raw dataset copy
├── outputs/                  # Generated metrics and visual artifacts
│   ├── figures/              # High-resolution (300 DPI) plots
│   │   ├── 01_class_distribution.png
│   │   ├── 02_sample_digits.png
│   │   ├── 03_mean_digit_heatmaps.png
│   │   ├── 04_pixel_intensity_distribution.png
│   │   ├── 05_confusion_matrices.png
│   │   ├── 06_model_comparison.png
│   │   ├── 07_misclassified_examples.png
│   │   └── 08_classwise_f1_scores.png
│   ├── test_evaluation_summary.csv
│   └── classwise_metrics_summary.csv
└── src/                      # Modular Python source package
    ├── __init__.py
    ├── data_loader.py        # Data ingestion, audit, and stratified split
    ├── eda.py                # Exploratory data analysis plots
    ├── models.py             # Preprocessing pipelines & CV tuning
    ├── evaluate.py           # Test set evaluation & error extraction
    └── visualize.py          # Diagnostic plots & error visualization
```

---

## 5. Summary of Final Results

### Benchmark Metrics (Untouched 20% Test Split & 5-Fold Cross-Validation)

| Model Family | 5-Fold CV Macro F1 (Mean +/- SD) | Test Accuracy | Test Accuracy 95% CI | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (Macro OVR) | Fit Time (s) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority)** | 0.0183 +/- 0.0000 | 10.00% | [7.2%, 13.6%] | 0.0100 | 0.1000 | 0.0182 | 0.5000 | 0.02 s | 0.001 ms |
| **Logistic Regression** | 0.9687 +/- 0.0084 | 95.83% | [93.2%, 97.5%] | 0.9585 | 0.9579 | 0.9579 | 0.9992 | 2.59 s | 0.002 ms |
| **Random Forest** | 0.9769 +/- 0.0076 | 96.39% | [93.9%, 97.9%] | 0.9647 | 0.9636 | 0.9635 | 0.9991 | 2.38 s | 0.031 ms |
| **Support Vector Machine** | **0.9881 +/- 0.0034** | **99.44%** | **[97.9%, 99.8%]** | **0.9946** | **0.9944** | **0.9944** | **0.99997** | **0.52 s** | **0.044 ms** |

### Per-Class Performance Table: Support Vector Machine (Test Split)

| Digit Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1.0000 | 1.0000 | 1.0000 | 36 |
| **1** | 0.9730 | 1.0000 | 0.9863 | 36 |
| **2** | 1.0000 | 1.0000 | 1.0000 | 35 |
| **3** | 1.0000 | 1.0000 | 1.0000 | 37 |
| **4** | 1.0000 | 1.0000 | 1.0000 | 36 |
| **5** | 1.0000 | 1.0000 | 1.0000 | 37 |
| **6** | 1.0000 | 1.0000 | 1.0000 | 36 |
| **7** | 0.9730 | 1.0000 | 0.9863 | 36 |
| **8** | 1.0000 | 0.9714 | 0.9855 | 35 |
| **9** | 1.0000 | 0.9722 | 0.9859 | 36 |
| **Macro Avg** | **0.9946** | **0.9944** | **0.9944** | **360** |

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
- [Full Project Report (PROJECT_REPORT.md)](file:///d:/Handwritten-digit-detection/PROJECT_REPORT.md)
- [Data Dictionary (data_dictionary.md)](file:///d:/Handwritten-digit-detection/data_dictionary.md)
