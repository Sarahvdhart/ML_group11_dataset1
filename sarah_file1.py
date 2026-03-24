# Main code for model training and evaluation

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

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
    outer_sens = []
    outer_spec = []
    roc_data[model_name] = []
    best_params_per_fold = []

    # Outer CV loop
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # RandomizedSearchCV
        grid = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,  
            n_iter=40,                           #nog beargumenteren                   
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

        #sensitivity and specificity
        y_pred_label = (y_pred >= 0.5).astype(int)

        # Confusion matrix voor sensitiviteit en specificiteit
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_label).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # recall voor klasse 1
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall voor klasse 0

        outer_sens.append(sensitivity)
        outer_spec.append(specificity)

        # Save best params for this fold
        best_params_per_fold.append(grid.best_params_)

        # ROC data opslaan
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_data[model_name].append((fpr, tpr))

        print(f"Fold {fold}: AUC = {auc:.4f}, Sens = {sensitivity:.3f}, Spec = {specificity:.3f}, Best params = {grid.best_params_}")

    all_results[model_name] = {
        "AUC": outer_scores,
        "Sensitivity": outer_sens,
        "Specificity": outer_spec
    }

# Print summary tabel
summary = []
for model, metrics in all_results.items():
    mean_auc = np.mean(metrics["AUC"])
    std_auc = np.std(metrics["AUC"])
    mean_sens = np.mean(metrics["Sensitivity"])
    mean_spec = np.mean(metrics["Specificity"])
    
    summary.append([model, mean_auc, std_auc, mean_sens, mean_spec])

df_summary = pd.DataFrame(summary, columns=["Model", "Mean AUC", "Std AUC", "Mean Sensitivity", "Mean Specificity"])
print("\n=== Summary Table ===")
print(df_summary.round(3))


# # Plot mean ROC curves per model
# plt.figure(figsize=(8, 6))
# for model_name, roc_folds in roc_data.items():
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)

#     for fpr, tpr in roc_folds:
#         # Interpoleer TPR naar dezelfde FPR-schaal
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(auc(fpr, tpr))

#     # Bereken gemiddelde en std TPR
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_auc = np.mean(aucs)
#     std_tpr = np.std(tprs, axis=0)
#     std_auc = np.std(aucs)

#     # Plot mean ROC
#     plt.plot(mean_fpr, mean_tpr,
#              label=f"{model_name} (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
#              lw=2)

#     # Plot ±1 std als shaded area
#     plt.fill_between(mean_fpr,
#                      mean_tpr - std_tpr,
#                      mean_tpr + std_tpr,
#                      alpha=0.2)

# # Chance line
# plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Mean ROC Curves per Classifier")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()


# # #boxplot met seaborn
# # # long format maken
# # df_plot = pd.DataFrame({
# #     "AUC": sum(all_results.values(), []),
# #     "Model": sum([[k]*len(v) for k, v in all_results.items()], [])
# # })

# # plt.figure()

# # sns.boxplot(x="Model", y="AUC", data=df_plot)
# # sns.stripplot(x="Model", y="AUC", data=df_plot, jitter=True)

# plt.title("Model comparison (Nested CV AUC)")
# plt.ylabel("AUC")

# plt.show()

# # ROC curve plotten
# plt.figure()

# for model_name, curves in roc_data.items():
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []

#     for fpr, tpr in curves:
#         tpr_interp = np.interp(mean_fpr, fpr, tpr)
#         tprs.append(tpr_interp)

#     mean_tpr = np.mean(tprs, axis=0)

#     plt.plot(mean_fpr, mean_tpr, label=model_name)

# # random classifier lijn
# plt.plot([0, 1], [0, 1], linestyle="--")

# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Mean ROC Curve (Nested CV)")
# plt.legend()

# plt.show()