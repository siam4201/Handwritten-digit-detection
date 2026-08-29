import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import FIGURES_DIR, IMAGE_SHAPE, CLASS_NAMES

# Setup Plot Aesthetics
def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Confusion Matrix Heatmaps
def plot_confusion_matrices(models_dict, predictions_dict, y_test):
    setup_plot_style()
    model_names = [m for m in models_dict.keys() if "Baseline" not in m]
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5), dpi=300)
    
    if len(model_names) == 1:
        axes = [axes]
        
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_test, predictions_dict[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("True Class", fontsize=9)
        
    plt.tight_layout()
    save_path = FIGURES_DIR / "05_confusion_matrices.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Model Comparison Summary Chart
def plot_model_comparison(summary_df, tuning_results):
    setup_plot_style()
    df = summary_df.copy()
    df["Train Time (s)"] = [tuning_results[m]["train_time_sec"] for m in df["Model"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    # Predictive Performance
    x = np.arange(len(df["Model"]))
    width = 0.35
    ax1.bar(x - width/2, df["Accuracy"], width, label="Accuracy", color="#1f77b4")
    ax1.bar(x + width/2, df["F1-Score (Macro)"], width, label="Macro F1", color="#ff7f0e")
    ax1.set_title("Test Predictive Metrics", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("Score", fontsize=9)
    ax1.legend(loc="upper left", frameon=True)
    
    # Computational Cost
    ax2.bar(df["Model"], df["Train Time (s)"], color="#2ca02c", width=0.5)
    ax2.set_title("Training Time / Computational Cost", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Time (seconds)", fontsize=9)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "06_model_comparison.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Diagnostic Visualization of Misclassified Samples
def plot_misclassified_samples(X_test, y_test, y_pred, model_name, max_samples=8):
    setup_plot_style()
    mismatches = np.where(y_test != y_pred)[0]
    
    if len(mismatches) == 0:
        return None
        
    n_display = min(len(mismatches), max_samples)
    fig, axes = plt.subplots(1, n_display, figsize=(2.2 * n_display, 2.8), dpi=300)
    
    if n_display == 1:
        axes = [axes]
        
    for i in range(n_display):
        idx = mismatches[i]
        img = X_test[idx].reshape(IMAGE_SHAPE)
        axes[i].imshow(img, cmap="gray_r", interpolation="nearest")
        axes[i].set_title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", fontsize=10, color="darkred")
        axes[i].axis("off")
        
    plt.suptitle(f"Error Diagnostic: Misclassified Cases ({model_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = FIGURES_DIR / "07_misclassified_examples.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Class-wise F1 Scores
def plot_classwise_f1(detailed_reports):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    
    models = [m for m in detailed_reports.keys() if "Baseline" not in m]
    width = 0.25
    x = np.arange(10)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, model in enumerate(models):
        f1_vals = [detailed_reports[model][str(c)]["f1-score"] for c in range(10)]
        ax.bar(x + (i - 1) * width, f1_vals, width, label=model, color=colors[i % len(colors)])
        
    ax.set_title("Class-wise F1-Score Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Digit Class", fontsize=10)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.75, 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "08_classwise_f1_scores.png"
    plt.savefig(save_path)
    plt.close()
    return save_path
