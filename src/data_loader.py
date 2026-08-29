import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from config import DATA_DIR, RANDOM_STATE, TEST_SIZE, CLASS_NAMES

# Data Loading
def load_raw_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    
    feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
    df = pd.DataFrame(digits.data, columns=feature_names)
    df["target"] = digits.target
    
    raw_csv_path = DATA_DIR / "raw_digits.csv"
    if not raw_csv_path.exists():
        df.to_csv(raw_csv_path, index=False)
        
    return digits.data, digits.target, digits.images, df

# Data Audit
def audit_data(X, y, df):
    audit_results = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_values": int(np.isnan(X).sum()),
        "min_pixel_value": float(np.min(X)),
        "max_pixel_value": float(np.max(X)),
        "n_duplicates": int(df.duplicated().sum()),
        "n_classes": len(np.unique(y)),
        "class_counts": {str(c): int(cnt) for c, cnt in zip(*np.unique(y, return_counts=True))}
    }
    return audit_results

# Leakage-Safe Stratified Splitting
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
