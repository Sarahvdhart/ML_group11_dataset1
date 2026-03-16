# import packages
from worclipo.load_data import load_data

# General packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# XGBoost
from xgboost import XGBClassifier


data = load_data()

# Selecting all features for feature selection within XGBoost
# CHANGE SO YOU DONT USE TEST DATA & NEED PRE PROCESSING
X = data.drop(columns=['label']).values  
y_raw = data['label'].values

# Encode string labels to numeric
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Train XGBClassifier on all features
clf = XGBClassifier()
clf.fit(X, y)

# Get top 2 features by importance
importance = clf.feature_importances_
top2_idx = np.argsort(importance)[-2:]
X_top2 = X[:, top2_idx]

# Retrain classifier on top 2 features for visualization
clf_top2 = XGBClassifier()
clf_top2.fit(X_top2, y)
y_pred_top2 = clf_top2.predict(X_top2)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Colorplot function
def colorplot(model, ax, X0, X1, resolution=100):
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)

# Scatter points and decision boundary
ax.scatter(X_top2[:, 0], X_top2[:, 1], c=y, s=25, edgecolor='k', cmap=plt.cm.Paired)
colorplot(clf_top2, ax, X_top2[:, 0], X_top2[:, 1])
ax.set_title(f"Top 2 features indices: {top2_idx}\nMisclassified: {(y != y_pred_top2).sum()} / {X_top2.shape[0]}")
plt.show()