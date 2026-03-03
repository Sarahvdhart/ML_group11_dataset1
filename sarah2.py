#importeren
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
from worclipo.load_data import load_data

data = load_data()

# #dataset stratified split in train en test
# from sklearn.model_selection import train_test_split

# #ID en label kolommen verwijderen voor de features
# X = data.drop(["ID", "label"], axis=1)

# y = data["label"]

# #nested cross validation
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


#jfdkjfk
# Imports
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Voorbeeld: data inladen
# data = pd.read_csv("your_dataset.csv")
# ID is index, label is target
X = data.drop("label", axis=1)
y = data["label"]

# Pipeline: imputatie + scaling + model
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Hyperparameters om te tunen
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

# Outer CV: voor performance estimate
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: voor hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV = inner CV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="accuracy"
)

# Nested CV: cross_val_score gebruikt outer CV
nested_scores = cross_val_score(
    grid_search,
    X,
    y,
    cv=outer_cv,
    scoring="accuracy"
)

print("Nested CV scores per fold:", nested_scores)
print("Mean accuracy:", nested_scores.mean())