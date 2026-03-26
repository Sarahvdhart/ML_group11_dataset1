#Import functions
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from preprocessing import CustomPreprocessor
from scipy.stats import loguniform, uniform, randint
import numpy as np

#Pipeline for XGBoost Classifier
def get_xgb_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=True,
            corr_threshold=0.85
        )),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ))
    ])

#Hyperparameter tuning
def get_xgb_param_grid():
    return {
        'classifier__n_estimators': np.arange(50,201),          # 50 to 200
        'classifier__max_depth': np.arange(3,8),               # 3 to 7
        'classifier__learning_rate': loguniform(0.01, 0.2),   # 0.01 to 0.2
        'classifier__subsample': uniform(0.7, 0.3),           # 0.7 to 1.0
        'classifier__colsample_bytree': uniform(0.7, 0.3),    # 0.7 to 1.0
        'classifier__reg_lambda': uniform(0.5, 1.0),          # 0.5 to 1.5
        'classifier__reg_alpha': uniform(0.0, 1.0)            # 0 to 1
    }
