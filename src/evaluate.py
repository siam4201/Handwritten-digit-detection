import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of execution method
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Test Set Evaluation
def evaluate_models(models_dict, X_test, y_test):
    metrics_summary = []
    predictions_dict = {}
    detailed_reports = {}
    classwise_tables = {}
    
    for name, model in models_dict.items():
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        
        # Predicted Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = np.nan
        else:
            roc_auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        
        metrics_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec_macro,
            "Recall (Macro)": rec_macro,
            "F1-Score (Macro)": f1_macro,
            "F1-Score (Weighted)": f1_weighted,
            "ROC-AUC (Macro OVR)": roc_auc,
            "Inference Latency (ms/sample)": inference_time
        })
        
        predictions_dict[name] = y_pred
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        detailed_reports[name] = report_dict
        
        # Build Per-Class Metrics DataFrame
        class_rows = []
        for digit in range(10):
            d_str = str(digit)
            if d_str in report_dict:
                class_rows.append({
                    "Digit": digit,
                    "Precision": report_dict[d_str]["precision"],
                    "Recall": report_dict[d_str]["recall"],
                    "F1-Score": report_dict[d_str]["f1-score"],
                    "Support": int(report_dict[d_str]["support"])
                })
        classwise_tables[name] = pd.DataFrame(class_rows)
        
    summary_df = pd.DataFrame(metrics_summary)
    return summary_df, predictions_dict, detailed_reports, classwise_tables

# Error Extraction for Diagnostic Analysis
def get_error_indices(y_true, y_pred):
    mismatches = np.where(y_true != y_pred)[0]
    error_data = []
    for idx in mismatches:
        error_data.append({
            "test_index": int(idx),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred[idx])
        })
    return error_data, mismatches

if __name__ == "__main__":
    from src.data_loader import load_raw_data, split_data
    from src.models import train_and_tune_models
    X, y, images, df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    trained_models, _ = train_and_tune_models(X_train, y_train)
    summary_df, predictions_dict, detailed_reports, classwise_tables = evaluate_models(trained_models, X_test, y_test)
    print("\n=== Model Evaluation Summary (Test Split) ===")
    print(summary_df.to_string(index=False))
