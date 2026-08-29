import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from config import RANDOM_STATE, CV_FOLDS

# Pipeline and Model Factory
def get_model_pipelines():
    pipelines = {
        "Baseline (Majority)": Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", SVC(probability=True, random_state=RANDOM_STATE))
        ])
    }
    return pipelines

# Hyperparameter Search Grids
def get_param_grids():
    grids = {
        "Baseline (Majority)": {},
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"]
        },
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20]
        },
        "Support Vector Machine": {
            "model__C": [0.5, 1.0, 5.0],
            "model__gamma": ["scale", "auto"]
        }
    }
    return grids

# Cross-Validation Training and Hyperparameter Tuning
def train_and_tune_models(X_train, y_train):
    pipelines = get_model_pipelines()
    param_grids = get_param_grids()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    trained_models = {}
    tuning_results = {}
    
    for name, pipeline in pipelines.items():
        grid = param_grids[name]
        start_time = time.time()
        
        if grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True
            )
            search.fit(X_train, y_train)
            fit_time = time.time() - start_time
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_
        else:
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - start_time
            best_model = pipeline
            best_params = {"strategy": "most_frequent"}
            best_cv_score = 0.0
            
        trained_models[name] = best_model
        tuning_results[name] = {
            "best_params": best_params,
            "best_cv_f1_macro": best_cv_score,
            "train_time_sec": fit_time
        }
        
    return trained_models, tuning_results
