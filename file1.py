#Main code for model training and evaluation

# Import libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from preprocessing import CustomPreprocessor
from SVM import get_svm_pipeline, get_svm_param_grid
# from rf import get_rf_pipeline, get_rf_param_grid
# from xgboost_model import get_xgb_pipeline, get_xgb_param_grid


# Load data
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

# Nested cross-validation setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# models to evaluate
models = {
    "SVM": (get_svm_pipeline(), get_svm_param_grid()), #nog definieren in svm.py
#     "Random Forest": (get_rf_pipeline(), get_rf_param_grid()), #nog definieren in rf.py
#     "XGBoost": (get_xgb_pipeline(), get_xgb_param_grid()) #nog definieren in xgb.py
} 

# for each model, perform nested cross-validation and print results
for model_name, (pipeline, param_grid) in models.items():
    grid = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=inner_cv, 
        scoring="roc_auc", 
        n_jobs=-1)
    scores = cross_val_score(
        grid, X, y, 
        cv=outer_cv, 
        scoring="roc_auc", 
        n_jobs=-1)
    print(f"{model_name} AUC: {scores.mean():.4f} ± {scores.std():.4f}")
