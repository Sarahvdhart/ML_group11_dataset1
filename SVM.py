#import functions
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import CustomPreprocessor
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from scipy.stats import loguniform, uniform

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
        "classifier__kernel": ["linear", "rbf", "poly", "sigmoid"],          # discrete
        "classifier__C": loguniform(0.01, 100),                              # continue op log-schaal
        "classifier__gamma": loguniform(0.001, 1),                           # continue op log-schaal
        "classifier__degree": [2, 3, 4],                                     # discrete
        "classifier__coef0": uniform(0.0, 1.0)                               # continue uniform 0–1
    })