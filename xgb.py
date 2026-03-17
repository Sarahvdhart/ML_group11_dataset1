from xgboost import XGBClassifier

def get_xgb_pipeline():
    return XGBClassifier(eval_metric='logloss')

def get_xgb_param_grid():
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }