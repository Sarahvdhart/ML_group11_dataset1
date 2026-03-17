import matplotlib.pyplot as plt 
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import worclipo.load_data 
from sklearn.pipeline import Pipeline
import preprocessing
from sklearn.feature_selection import SelectFdr, VarianceThreshold, f_classif, SelectKBest

#------------------------------------------------------------------------------------------------
#Defining the pipeline for Random Forest Classifier
def get_rf_pipeline():
    return Pipeline([
        ("preprocess", preprocessing.CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("feature_selection", SelectFdr(score_func=f_classif, alpha=0.1)),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
#------------------------------------------------------------------------------------------------
# Defining the hyperparameter grid for Random Forest Classifier
def get_rf_param_grid():
    return {
        'n_estimators': [100, 200, 500], #amount of trees
        'max_depth': [3, 4, 5, 6], #aantal splitsingen
        'min_samples_split': [5, 10, 15],#minimum amount of samples for a split in a tree
        'min_samples_leaf': [2, 4, 8], #minimum amount of samples for a leaf in a tree
        'max_features': ['sqrt', 'log2', 0.1, 0.2, 0.3], # 0.1 t/m 0.3 per decision, use only a subset of features for each tree
        'criterion': ['gini', 'log_loss'], #splitting criterion
        'max_samples': [None, 0.7, 0.8], #dont use all samples for each tree, use a subset of samples for each tree: bootstrap sampling
        'ccp_alpha': [0.0, 0.01, 0.02] #how aggresive do you want to prune the tree (i.e. remove branches that have little importance)
        }
#------------------------------------------------------------------------------------------------
