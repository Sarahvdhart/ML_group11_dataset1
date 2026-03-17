# ook meteen printen
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
from sklearn.metrics import roc_auc_score #toegevoegd
from SVM import get_svm_pipeline, get_svm_param_grid
#from RF import get_rf_pipeline, get_rf_param_grid
#from XGB import get_xgb_pipeline, get_xgb_param_grid


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
 #    "XGBoost": (get_xgb_pipeline(), get_xgb_param_grid()) #nog definieren in xgb.py
} 

# # for each model, perform nested cross-validation and print results
# for model_name, (pipeline, param_grid) in models.items():
#     grid = GridSearchCV(
#         estimator=pipeline, 
#         param_grid=param_grid, 
#         cv=inner_cv, 
#         scoring="roc_auc", 
#         n_jobs=-1)
#     scores = cross_val_score(
#         grid, X, y, 
#         cv=outer_cv, 
#         scoring="roc_auc", 
#         n_jobs=-1)
#     print(f"{model_name} AUC: {scores.mean():.4f} ± {scores.std():.4f}")


# Loop over models
for model_name, (pipeline, param_grid) in models.items():
    print(f"\n===== {model_name} =====")
    
    outer_scores = []
    best_params_per_fold = []

    # Outer CV loop
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV for hyperparameter tuning
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # Test best model on outer fold
        best_model = grid.best_estimator_
        if hasattr(best_model, "decision_function"):
            y_pred = best_model.decision_function(X_test)
        else:
            y_pred = best_model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred)
        outer_scores.append(auc)
        best_params_per_fold.append(grid.best_params_)

        print(f"Fold {fold}: AUC = {auc:.4f}, Best params = {grid.best_params_}")

    # Print overall result
    print(f"\n{model_name} Mean AUC: {pd.Series(outer_scores).mean():.4f} ± {pd.Series(outer_scores).std():.4f}")
   
