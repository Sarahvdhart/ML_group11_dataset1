# General packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


#data importeren
from worclipo.load_data import load_data
data = load_data()

# Alleen numerieke kolommen selecteren
numeric_data = data.select_dtypes(include="number") 

# Data schalen met robust scaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(numeric_data)

# PCA toepassen
pca = PCA(n_components=2)  # Aantal componenten kiezen 
pca_result = pca.fit_transform(scaled_data)

# Labels handmatig coderen
label_order = ["lipoma", "liposarcoma"]  # bepaal zelf de volgorde
le = LabelEncoder()
le.fit(label_order)  # fit op deze volgorde

# Transformeer de labels van je dataset
y_numeric = le.transform(data["label"])

# PCA-resultaten in een DataFrame zetten
pca_df = pd.DataFrame(data=pca_result, columns=["Principal Component 1", "Principal Component 2"])
pca_df["Label"] = y_numeric # Label toevoegen voor visualisatie

# Classifier trainen op de volledige dataset
clf = LogisticRegression()
clf.fit(pca_result, y_numeric)


# Functie om decision boundary te plotten
def plot_decision_boundary(clf, X, y, ax):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = ax.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    return scatter

# Plotten
fig, ax = plt.subplots(figsize=(8,6))
scatter = plot_decision_boundary(clf, pca_result, y_numeric, ax)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("PCA + Decision Boundary (full dataset)")

# Legend met originele labels
handles, _ = scatter.legend_elements()
labels = le.inverse_transform(np.arange(len(le.classes_)))
ax.legend(handles, labels, title="Label")

plt.show()

print(pca.explained_variance_ratio_)