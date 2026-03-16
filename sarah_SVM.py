# SVM proberen

#importeren
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC
from preprocessing import CustomPreprocessor

#data inladen
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

#preprocessing
preprocessor = CustomPreprocessor(nan_threshold=0.30, zero_threshold=0.95, clip_iqr=False)
X_processed = preprocessor.fit_transform(X)

# print de grootte van de dataset na preprocessing
print("Grootte van de dataset na preprocessing:", X_processed.shape)

#nested innerloop
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# pipelines voor Linear SVM en RBF SVM

# Linear SVM
linear_pipeline = Pipeline([
    ("svm", SVC(kernel="linear"))
])

# RBF SVM
rbf_pipeline = Pipeline([
    ("svm", SVC(kernel="rbf"))
])


# Cross-validation scores
scores_linear = cross_val_score(linear_pipeline, X_processed, y, cv=inner_cv, scoring="roc_auc")
scores_rbf = cross_val_score(rbf_pipeline, X_processed, y, cv=inner_cv, scoring="roc_auc")

# Print resultaten
print("Linear SVM CV scores:", scores_linear)
print("Mean Linear SVM CV score:", scores_linear.mean())

print("RBF SVM CV scores:", scores_rbf)
print("Mean RBF SVM CV score:", scores_rbf.mean())
