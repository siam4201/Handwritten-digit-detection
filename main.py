import sys
import pandas as pd
from config import OUTPUT_DIR, FIGURES_DIR, MODELS_DIR
from src.data_loader import load_raw_data, audit_data, split_data
from src.eda import (
    plot_class_distribution,
    plot_sample_digits,
    plot_mean_digit_heatmaps,
    plot_pixel_distribution
)
from src.models import train_and_tune_models
from src.evaluate import evaluate_models, get_error_indices
from src.visualize import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_misclassified_samples,
    plot_classwise_f1
)

def run_pipeline():
    print("=== Step 1: Data Loading & Audit ===")
    X, y, images, df = load_raw_data()
    audit = audit_data(X, y, df)
    print(f"Loaded {audit['n_samples']} samples, {audit['n_features']} features, {audit['n_classes']} classes.")
    print(f"Missing values: {audit['missing_values']}, Duplicates: {audit['n_duplicates']}")
    
    print("\n=== Step 2: Stratified Train/Test Split ===")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")
    
    print("\n=== Step 3: Exploratory Data Analysis ===")
    plot_class_distribution(y_train, y_test)
    plot_sample_digits(images, y)
    plot_mean_digit_heatmaps(X_train, y_train)
    plot_pixel_distribution(X_train)
    print(f"Saved EDA figures to {FIGURES_DIR}")
    
    print("\n=== Step 4: Model Training & Hyperparameter Tuning ===")
    trained_models, tuning_results = train_and_tune_models(X_train, y_train)
    for model_name, info in tuning_results.items():
        print(f"[{model_name}] Best CV F1: {info['best_cv_f1_macro']:.4f} | Time: {info['train_time_sec']:.2f}s | Params: {info['best_params']}")
        
    print("\n=== Step 5: Untouched Test Evaluation ===")
    summary_df, predictions_dict, detailed_reports = evaluate_models(trained_models, X_test, y_test)
    print("\n" + summary_df.to_string(index=False))
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv_path = OUTPUT_DIR / "test_evaluation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved metrics summary to {summary_csv_path}")
    
    print("\n=== Step 6: Diagnostic Visualization & Error Analysis ===")
    plot_confusion_matrices(trained_models, predictions_dict, y_test)
    plot_model_comparison(summary_df, tuning_results)
    plot_classwise_f1(detailed_reports)
    
    best_model_name = "Support Vector Machine"
    best_preds = predictions_dict[best_model_name]
    error_list, _ = get_error_indices(y_test, best_preds)
    print(f"Total test errors for {best_model_name}: {len(error_list)} / {len(y_test)}")
    plot_misclassified_samples(X_test, y_test, best_preds, best_model_name)
    
    print(f"Saved diagnostic plots to {FIGURES_DIR}")
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
