# XGBoost
# import packages
from worclipo.load_data import load_data
from preprocessing import CustomPreprocessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# data inladen
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

start_time = time.time()  # start timer

def get_xgb_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("classifier", xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ))
    ])

# Roep pipeline op
pipeline = get_xgb_pipeline() 

# Hyperparameter tuning

param_grid = {
    "classifier__max_depth": [3, 5, 7],
    "classifier__n_estimators": [50, 100, 150],
    "classifier__learning_rate": [0.1, 0.01],
    "classifier__subsample": [0.5, 0.7, 1.0],
    "classifier__colsample_bytree": [0.7, 1.0]
}

# Hyperparamater grid
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,                # 5-fold cross-validation
    scoring="accuracy",  
    n_jobs=-1,
    verbose=2
)

# === Fit the grid search (trains pipeline with all parameter combos) ===
grid.fit(X, y)

# === Access the best model and parameters ===
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

end_time = time.time()    # end timer
total_time = end_time - start_time

print(f"Total runtime: {total_time:.2f} seconds")

