# Main code for model training and evaluation

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve 
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.linear_model import LogisticRegression
# from preprocessing import CustomPreprocessor



#import classifiers and the param grids
from SVM import get_svm_pipeline, get_svm_param_grid
from RF import get_rf_pipeline, get_rf_param_grid
from xgb import get_xgb_pipeline, get_xgb_param_grid

# Load data
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1})
X = df.drop(columns=["ID", "label"])

# Nested cross-validation setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# models to evaluate
models = {
    "SVM": (get_svm_pipeline(), get_svm_param_grid()), 
    "Random Forest": (get_rf_pipeline(), get_rf_param_grid()),
    "XGBoost": (get_xgb_pipeline(), get_xgb_param_grid()) 
} 

all_results = {}
roc_data = {}

# Loop over models
for model_name, (pipeline, param_grid) in models.items():
    print(f"\n===== {model_name} =====")
    
    outer_scores = []
    roc_data[model_name] = []
    summary = []
    best_params_per_fold = []


    # Outer CV loop
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # RandomizedSearchCV
        grid = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,  
            n_iter=40,  #nog beargumenteren                   
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42 
        )

        grid.fit(X_train, y_train)

        # Test best model on outer fold
        best_model = grid.best_estimator_
        if hasattr(best_model, "decision_function"):
            y_pred = best_model.decision_function(X_test)
        else:
            y_pred = best_model.predict_proba(X_test)[:, 1]

        #auc
        auc = roc_auc_score(y_test, y_pred)
        outer_scores.append(auc)

        best_params_per_fold.append(grid.best_params_)

        # ROC data opslaan
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_data[model_name].append((fpr, tpr))

        print(f"Fold {fold}: AUC = {auc:.4f}, Best params = {grid.best_params_}")

    all_results[model_name] = outer_scores

    # Print overall result
    print(f"\n{model_name} Mean AUC: {pd.Series(outer_scores).mean():.4f} ± {pd.Series(outer_scores).std():.4f}")

#print tabel
for model, scores in all_results.items():
    mean_auc = pd.Series(scores).mean()
    std_auc = pd.Series(scores).std()
    
    summary.append([model, mean_auc, std_auc])

df_summary = pd.DataFrame(summary, columns=["Model", "Mean AUC", "Std"])

print("\n=== Summary Table ===")
print(df_summary.round(3))

# # data voorbereiden
# data = [all_results["SVM"], 
#         all_results["Random Forest"], 
#         all_results["XGBoost"]]

# labels = ["SVM", "Random Forest", "XGBoost"]

# # boxplot
# plt.figure()
# plt.boxplot(data, labels=labels)

# plt.ylabel("AUC")
# #voeg mean en std toe aan de titel
# mean_std = {model: f"{pd.Series(scores).mean():.4f} ± {pd.Series(scores).std():.4f}" for model, scores in all_results.items()}
# plt.title(f"Model comparison (Nested CV AUC)\n" + "\n".join([f"{model}: {mean_std[model]}" for model in labels]))
# plt.title("Model comparison (Nested CV AUC)")
# plt.show()

#boxplot met seaborn
# long format maken
df_plot = pd.DataFrame({
    "AUC": sum(all_results.values(), []),
    "Model": sum([[k]*len(v) for k, v in all_results.items()], [])
})

plt.figure()

sns.boxplot(x="Model", y="AUC", data=df_plot)
sns.stripplot(x="Model", y="AUC", data=df_plot, jitter=True)

plt.title("Model comparison (Nested CV AUC)")
plt.ylabel("AUC")

plt.show()

# ROC curve plotten
plt.figure()

for model_name, curves in roc_data.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fpr, tpr in curves:
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)

    plt.plot(mean_fpr, mean_tpr, label=model_name)

# random classifier lijn
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve (Nested CV)")
plt.legend()

plt.show()