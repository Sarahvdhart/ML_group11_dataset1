#importeren van de nodige libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from preprocessing import CustomPreprocessor

# Data inlezen
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

# Nested CV instellen
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Pipeline
pipeline = Pipeline([
    ("preprocess", CustomPreprocessor(
        nan_threshold=0.30,
        zero_threshold=0.95,
        clip_iqr=False
    )),
    ("feature_selection", SelectKBest(score_func=f_classif)),
    ("classifier", LogisticRegression(max_iter=2000, solver="liblinear"))
])

## testen
import pandas as pd

df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")

y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

pre = CustomPreprocessor(
    nan_threshold=0.30,
    zero_threshold=0.95,
    clip_iqr=False
)

pre.fit(X)

X_clean = pre.transform(X)

print("Original shape:", X.shape)
print("Processed shape:", X_clean.shape)

import numpy as np

print("NaNs after preprocessing:", np.isnan(X_clean).sum())


