#Code that contains pipeline and hyperparameter grid for SVM

#Import functions
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import CustomPreprocessor
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from scipy.stats import loguniform, uniform

#Pipeline for SVM
def get_svm_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=True, 
            corr_threshold=0.85
        )),
        ("fold_variance_filter", VarianceThreshold(threshold=0)), #Remove features with 0 variance
        ("feature_selection", SelectKBest(score_func=f_classif, k=20)), #Select best 20 features
        ("classifier", SVC(
        ))
    ])

#Hyperparametergrid SVM
def get_svm_param_grid():
    return ({
        "classifier__kernel": ["linear", "rbf", "poly", "sigmoid"],         
        "classifier__C": loguniform(0.01, 100),                              
        "classifier__gamma": loguniform(0.001, 1),                          
        "classifier__degree": [2, 3, 4],                                    
        "classifier__coef0": uniform(0.0, 1.0)                               
    })