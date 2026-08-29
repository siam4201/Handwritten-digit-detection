import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import FIGURES_DIR, IMAGE_SHAPE, CLASS_NAMES

# Setup Plot Aesthetics
def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Exploratory Analysis: Class Distribution
def plot_class_distribution(y_train, y_test):
    setup_plot_style()
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    x = np.arange(len(train_classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.bar(x - width/2, train_counts, width, label="Train Split (80%)", color="#2b5c8f")
    ax.bar(x + width/2, test_counts, width, label="Test Split (20%)", color="#d95f02")

    ax.set_title("Class Distribution Across Stratified Splits", fontsize=12, fontweight="bold")
    ax.set_xlabel("Digit Class", fontsize=10)
    ax.set_ylabel("Sample Count", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, max(train_counts) * 1.25)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Exploratory Analysis: Sample Digits Grid
def plot_sample_digits(images, labels):
    setup_plot_style()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5), dpi=300)
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        idx = np.where(labels == i)[0][0]
        ax.imshow(images[idx], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"Class: {i}", fontsize=11)
        ax.axis("off")
        
    plt.suptitle("Representative Handwritten Digit Samples (8x8)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "02_sample_digits.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Exploratory Analysis: Mean Pixel Intensity per Class
def plot_mean_digit_heatmaps(X_train, y_train):
    setup_plot_style()
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5), dpi=300)
    
    for i in range(10):
        ax = axes[i // 5, i % 5]
        class_samples = X_train[y_train == i]
        mean_img = class_samples.mean(axis=0).reshape(IMAGE_SHAPE)
        sns.heatmap(mean_img, ax=ax, cmap="magma", cbar=False, square=True)
        ax.set_title(f"Mean Digit '{i}'", fontsize=10)
        ax.axis("off")
        
    plt.suptitle("Average Class Intensity Profiles (Training Split)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "03_mean_digit_heatmaps.png"
    plt.savefig(save_path)
    plt.close()
    return save_path

# Exploratory Analysis: Pixel Intensity Distribution
def plot_pixel_distribution(X_train):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    sns.histplot(X_train.flatten(), bins=17, discrete=True, color="#4daf4a", ax=ax, edgecolor="black")
    ax.set_title("Distribution of Raw Pixel Intensities [0-16]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel Value", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / "04_pixel_intensity_distribution.png"
    plt.savefig(save_path)
    plt.close()
    return save_path
