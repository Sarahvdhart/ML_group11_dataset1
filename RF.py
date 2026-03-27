#Code that contains pipeline and hyperparameter grid for RF

#Import functions
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import preprocessing

#Pipeline for RF
def get_rf_pipeline():
    return Pipeline([
        ("preprocess", preprocessing.CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=True,
            corr_threshold=0.85
        )),
        ("classifier", RandomForestClassifier(
            random_state=42
            ))
    ])

#Hyperparameter grid RF
def get_rf_param_grid():
    return {
        'classifier__n_estimators': list(range(100, 401, 5)),                   
        'classifier__max_depth': [3, 4, 5, 6], 
        'classifier__min_samples_split': list(range(5, 16)), 
        'classifier__min_samples_leaf': [3, 4, 5, 6, 7, 8], 
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.3], 
        'classifier__criterion': ['gini', 'log_loss'], 
        'classifier__max_samples': [None, 0.7, 0.8], 
        'classifier__ccp_alpha': np.arange(0, 0.022, 0.002).tolist(), 
        'classifier__bootstrap': [True]
        }
