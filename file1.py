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




## Test if preprocessing works correctly
# import pandas as pd

# df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")

# y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
# X = df.drop(columns=["ID", "label"])

# pre = CustomPreprocessor(zero_threshold=0.90, clip_iqr=False, corr_threshold=0.85)

# pre.fit(X)

# X_clean = pre.transform(X)

# print("Original shape:", X.shape)
# print("Processed shape:", X_clean.shape)
# pre = CustomPreprocessor(zero_threshold=0.90, clip_iqr=False, corr_threshold=0.85)
# X_clean = pre.fit_transform(X)


# # testen hoeveel features er overblijven na preprocessing en feature selectie
# pipeline.fit(X, y)

# preprocessor = pipeline.named_steps["preprocess"]
# selector = pipeline.named_steps["feature_selection"]

# X_processed = preprocessor.transform(X)

# print("Features origineel:", X.shape[1])
# print("Features na preprocessing:", X_processed.shape[1])
# print("Features na feature selectie:", selector.get_support().sum())


## test 2
# df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")

# y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
# X = df.drop(columns=["ID", "label"])

# print("Aantal samples:", X.shape[0])
# print("Aantal features vóór preprocessing:", X.shape[1])

# ## test 3
# import pandas as pd
# import numpy as np

# df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")

# y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
# X = df.drop(columns=["ID", "label"])

# print("Start aantal features:", X.shape[1])

# # -------------------------
# # 1. Zero filter
# # -------------------------
# zero_ratio = (X == 0).mean()
# keep_zero = zero_ratio <= 0.90

# removed_zero = X.shape[1] - keep_zero.sum()

# X_zero = X.loc[:, keep_zero]

# print("Removed by zero filter:", removed_zero)
# print("Remaining after zero filter:", X_zero.shape[1])

# # -------------------------
# # 2. Variance filter
# # -------------------------
# variance = X_zero.var()
# keep_var = variance > 0

# removed_var = X_zero.shape[1] - keep_var.sum()

# X_var = X_zero.loc[:, keep_var]

# print("Removed by variance=0:", removed_var)
# print("Remaining after variance filter:", X_var.shape[1])

# # -------------------------
# # 3. IQR clipping (geen features verwijderd)
# # -------------------------
# Q1 = X_var.quantile(0.25)
# Q3 = X_var.quantile(0.75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# print("IQR clipping removes 0 features (only clips values)")

# # -------------------------
# # 4. Correlation filter
# # -------------------------
# corr_matrix = X_var.corr().abs()

# upper = corr_matrix.where(
#     np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
# )

# high_corr_features = [col for col in upper.columns if any(upper[col] > 0.85)]

# X_corr = X_var.drop(columns=high_corr_features)

# print("Removed by correlation filter:", len(high_corr_features))
# print("Remaining after correlation filter:", X_corr.shape[1])

# # -------------------------
# # Final result
# # -------------------------
# print("\nFinal number of features:", X_corr.shape[1])


