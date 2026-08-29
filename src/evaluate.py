import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Test Set Evaluation
def evaluate_models(models_dict, X_test, y_test):
    metrics_summary = []
    predictions_dict = {}
    detailed_reports = {}
    
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
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
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
        detailed_reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
    summary_df = pd.DataFrame(metrics_summary)
    return summary_df, predictions_dict, detailed_reports

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
