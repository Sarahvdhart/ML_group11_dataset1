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
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
