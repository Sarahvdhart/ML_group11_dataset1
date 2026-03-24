import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import worclipo.load_data 
from sklearn.pipeline import Pipeline
import preprocessing
from sklearn.feature_selection import SelectFdr, VarianceThreshold, f_classif, SelectKBest

#-------------------------------------------------------------------------------------------------
#Defining the pipeline for Random Forest Classifier
def get_rf_pipeline():
    return Pipeline([
        ("preprocess", preprocessing.CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=True,
            corr_threshold=0.85
        )),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
#------------------------------------------------------------------------------------------------
# Defining the hyperparameter grid for Random Forest Classifier
# def get_rf_param_grid():
#     return {
#         'classifier__n_estimators': [100, 500], #amount of trees
#         'classifier__max_depth': [3,6], #aantal splitsingen
#         'classifier__min_samples_split': [5],#minimum amount of samples for a split in a tree
#         'classifier__min_samples_leaf': [4, 8], #minimum amount of samples for a leaf in a tree
#         'classifier__max_features': ['sqrt', 'log2', 0.3], # 0.1 t/m 0.3 per decision, use only a subset of features for each tree
#         'classifier__criterion': ['gini', 'log_loss'], #splitting criterion
#         'classifier__max_samples': [None, 0.7, 0.8], #dont use all samples for each tree, use a subset of samples for each tree: bootstrap sampling
#         'classifier__ccp_alpha': [0.0, 0.02] #how aggresive do you want to prune the tree (i.e. remove branches that have little importance)
#         }
#------------------------------------------------------------------------------------------------
# Defining the hyperparameter grid for Random Forest Classifier
def get_rf_param_grid():
    return {
        'classifier__n_estimators': [100, 200, 300, 400], #amount of trees
        'classifier__max_depth': [3, 4, 5, 6], #total amount of splits allowed in a tree; how deep the tree can grow
        'classifier__min_samples_split': list(range(5, 16)), #minimum amount of samples for a split in a tree
        'classifier__min_samples_leaf': [3, 5, 8], #minimum amount of samples for a leaf in a tree
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.3], # 0.1 t/m 0.3 per decision, use only a subset of features for each tree
        'classifier__criterion': ['gini', 'log_loss'], #splitting criterion
        'classifier__max_samples': [None, 0.7, 0.8], #dont use all samples for each tree, use a subset of samples for each tree: bootstrap sampling
        'classifier__ccp_alpha': [0.0, 0.01, 0.02], #how aggresive do you want to prune the tree (i.e. remove branches that have little importance)
        'classifier__bootstrap': [True]
        }
#------------------------------------------------------------------------------------------------