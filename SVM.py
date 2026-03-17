#import functions
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import CustomPreprocessor
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest


#pipeline for SVM
def get_svm_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("fold_variance_filter", VarianceThreshold(threshold=0)), #haalt constante features eruit vlak voor anova
        ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
        ("classifier", SVC())
    ])

#hyperparameter tuning
def get_svm_param_grid():
    return ({        
        "classifier__kernel": ["linear", "rbf"],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1]  # only relevant for rbf
    }) 