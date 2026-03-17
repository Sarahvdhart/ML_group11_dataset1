from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from preprocessing import CustomPreprocessor

def get_xgb_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ))
    ])

def get_xgb_param_grid():
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
