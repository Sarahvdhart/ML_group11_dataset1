# Main code for model training and evaluation

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score

#import classifiers and the param grids
from SVM import get_svm_pipeline, get_svm_param_grid
from RF import get_rf_pipeline, get_rf_param_grid
from xgb import get_xgb_pipeline, get_xgb_param_grid

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

best_params_all_models = {}

# Loop over models
for model_name, (pipeline, param_grid) in models.items():
    print(f"\n===== {model_name} =====")
    
    outer_scores = []
    outer_acc = []
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

        # Test best model on outer fold for AUC
        best_model = grid.best_estimator_
        if hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        else:
            y_score = best_model.predict_proba(X_test)[:, 1]

        # voor accuracy/sensitivity/specificity
        y_pred = best_model.predict(X_test)
        
        #auc
        fold_auc = roc_auc_score(y_test, y_score)
        outer_scores.append(fold_auc)

        #accuracy
        accuracy = accuracy_score(y_test, y_pred)
        outer_acc.append(accuracy)

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
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[model_name].append((fpr, tpr))

        print(
            f"Fold {fold}: AUC = {fold_auc:.4f}, Acc = {accuracy:.3f}, "
            f"Sens = {sensitivity:.3f}, Spec = {specificity:.3f}, "
            f"Best params = {grid.best_params_}"
        )

    all_results[model_name] = {
        "AUC": outer_scores,
        "Accuracy": outer_acc,
        "Sensitivity": outer_sens,
        "Specificity": outer_spec
    }

    best_params_all_models[model_name] = best_params_per_fold

# Print summary tabel
summary = []
for model, metrics in all_results.items():
    mean_auc = np.mean(metrics["AUC"])
    std_auc = np.std(metrics["AUC"])
    mean_acc = np.mean(metrics["Accuracy"])
    std_acc = np.std(metrics["Accuracy"])
    mean_sens = np.mean(metrics["Sensitivity"])
    std_sens = np.std(metrics["Sensitivity"])
    mean_spec = np.mean(metrics["Specificity"])
    std_spec = np.std(metrics["Specificity"])

    
    summary.append([model, mean_auc, std_auc, mean_acc, std_acc,mean_sens, std_sens, mean_spec, std_spec])

df_summary = pd.DataFrame(summary, columns=["Model", "Mean AUC", "Std AUC", "Mean Accuracy", "Std Accuracy", "Mean Sensitivity", "Std Sensitivity", "Mean Specificity", "Std Specificity"])
print("\n=== Summary Table ===")
print(df_summary.round(3))


# Plot ROC curves in 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, roc_folds) in zip(axes, roc_data.items()):
    tprs = [] #true positive rate
    aucs_per_fold = []
    mean_fpr = np.linspace(0, 1, 100) #false positive rate

    for fpr, tpr in roc_folds:
        # Interpoleer TPR naar dezelfde FPR-schaal
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs_per_fold.append(auc(fpr, tpr))

    # Bereken gemiddelde en std TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs_per_fold)
    std_tpr = np.std(tprs, axis=0)
    std_auc = np.std(aucs_per_fold)

    # Plot mean ROC
    ax.plot(mean_fpr, mean_tpr,
            label=f"{model_name}\nAUC = {mean_auc:.2f} ± {std_auc:.2f}",
            lw=2)

    # Plot ±1 std als shaded area
    ax.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    alpha=0.2)

    # Chance line
    ax.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(model_name)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

plt.suptitle("Mean ROC Curves per Classifier", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print best classifier
best_model_name = df_summary.loc[df_summary["Mean AUC"].idxmax(), "Model"]
print(f"\nBest model based on AUC: {best_model_name}")

from collections import Counter
import numpy as np

def select_final_hyperparameters(best_params_per_fold, fold_scores):
    final_params = {}
    param_names = best_params_per_fold[0].keys()

    for param in param_names:
        values = [params[param] for params in best_params_per_fold]

        counts = Counter(values)
        max_count = max(counts.values())
        candidates = [val for val, count in counts.items() if count == max_count]

        # als één duidelijke winnaar
        if len(candidates) == 1:
            final_params[param] = candidates[0]
        else:
            # tie-break → beste gemiddelde performance
            candidate_perf = {}

            for candidate in candidates:
                scores = [
                    fold_scores[i]
                    for i in range(len(best_params_per_fold))
                    if best_params_per_fold[i][param] == candidate
                ]
                candidate_perf[candidate] = np.mean(scores)

            best_candidate = max(candidate_perf, key=candidate_perf.get)
            final_params[param] = best_candidate

    return final_params

final_best_params = select_final_hyperparameters(
    best_params_all_models[best_model_name],
    all_results[best_model_name]["AUC"]
)

print("\nFinal selected hyperparameters:")
for k, v in final_best_params.items():
    print(f"{k}: {v}")

best_pipeline, _ = models[best_model_name]

final_model = best_pipeline.set_params(**final_best_params)
final_model.fit(X, y)

print(f"\nFinal {best_model_name} model trained on all data.")   
