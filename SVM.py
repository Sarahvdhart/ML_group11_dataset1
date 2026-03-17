#import functions
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import CustomPreprocessor
from sklearn.feature_selection import SelectFdr, f_classif


#-----------------------------------------------------------------------------------------------
#pipeline for SVM
def get_svm_pipeline():
    return Pipeline([
        ("preprocess", CustomPreprocessor(
            zero_threshold=0.90,
            clip_iqr=False,
            corr_threshold=0.85
        )),
        ("feature_selection", SelectFdr(score_func=f_classif, alpha=0.1)),
        ("classifier", SVC())
    ])

#------------------------------------------------------------------------------------------------------------
# #data inladen
# df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
# y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
# X = df.drop(columns=["ID", "label"])

#---------------------------------------------------------------------------------------------------------------
#maar hiervoor moet nog de outerloop worden gesplitst
#nested innerloop
# inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

#hyperparameter tuning
def get_svm_param_grid():
    return ({        
        "classifier__kernel": ["linear", "rbf"],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1]  # only relevant for rbf
    }) 

# grid_search = GridSearchCV(
#     estimator=get_svm_pipeline(), 
#     param_grid=get_svm_param_grid, 
#     cv=inner_cv, 
#     scoring="roc_auc",
#     verbose=2,
#     n_jobs=-1
#     )

# grid_search.fit(X, y)

# -----------------------------------------------------------------------------------------------
# # resultaten
# print("Beste parameters:", grid_search.best_params_)
# print("Beste ROC-AUC:", grid_search.best_score_)

