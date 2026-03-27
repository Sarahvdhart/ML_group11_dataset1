#Main code for model training and evaluation

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score

#Import classifiers and the param grids
from SVM import get_svm_pipeline, get_svm_param_grid
from RF import get_rf_pipeline, get_rf_param_grid
from xgb import get_xgb_pipeline, get_xgb_param_grid

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Load data
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
y = df["label"].map({"lipoma": 0, "liposarcoma": 1}) 
X = df.drop(columns=["ID", "label"])

#Nested cross-validation setup
#StratifiedKfold to preserve class distribution
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Set random state for reproducibility 
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

#List of models to evaluate
models = {
    "SVM": (get_svm_pipeline(), get_svm_param_grid()),              
    "Random Forest": (get_rf_pipeline(), get_rf_param_grid()),
    "XGBoost": (get_xgb_pipeline(), get_xgb_param_grid()) 
} 

all_results = {}
roc_data = {}

best_params_all_models = {}

#Loop over classifiers: SVM, RF & XGBoost
for model_name, (pipeline, param_grid) in models.items():
    print(f"\n===== {model_name} =====")
    
    outer_scores = []
    outer_acc = []
    roc_data[model_name] = []
    best_params_per_fold = []

    #Outer nested CV loop
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        #RandomizedSearchCV for shorter computation time
        grid = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,  
            n_iter=40,                         #40 iterations chosen for balance between speed and grid coverage                   
            cv=inner_cv,
            scoring="roc_auc",                 #ROC_AUC provides single score to compare classifiers        
            n_jobs=-1,
            random_state=42 
        )

        grid.fit(X_train, y_train)

        #Test best model on outer fold for AUC
        best_model = grid.best_estimator_
        if hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_test)
        else:
            y_score = best_model.predict_proba(X_test)[:, 1]

        y_pred = best_model.predict(X_test)
        
        #Determine AUC for classifier selection
        fold_auc = roc_auc_score(y_test, y_score)
        outer_scores.append(fold_auc)

        #Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        outer_acc.append(accuracy)

        #Save ROC data for figure
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[model_name].append((fpr, tpr))

        print(
            f"Fold {fold}: AUC = {fold_auc:.4f}, Acc = {accuracy:.3f}, "
            f"Best params = {grid.best_params_}"
        )

    all_results[model_name] = {
        "AUC": outer_scores,
        "Accuracy": outer_acc,
    }

    best_params_all_models[model_name] = best_params_per_fold

#Print summary tabel to compare classifiers
summary = []
for model, metrics in all_results.items():
    mean_auc = np.mean(metrics["AUC"])
    std_auc = np.std(metrics["AUC"])
    mean_acc = np.mean(metrics["Accuracy"])
    std_acc = np.std(metrics["Accuracy"])
    
    summary.append([model, mean_auc, std_auc, mean_acc, std_acc])

df_summary = pd.DataFrame(summary, columns=["Model", "Mean AUC", "Std AUC", "Mean Accuracy", "Std Accuracy"])
print("\n=== Summary Table ===")
print(df_summary.round(3))


#Plot ROC curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, roc_folds) in zip(axes, roc_data.items()):
    tprs = [] #True positive rate
    aucs_per_fold = []
    mean_fpr = np.linspace(0, 1, 100) #False positive rate

    for fpr, tpr in roc_folds:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs_per_fold.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs_per_fold)
    std_tpr = np.std(tprs, axis=0)
    std_auc = np.std(aucs_per_fold)

    ax.plot(mean_fpr, mean_tpr,
            label=f"{model_name}\nAUC = {mean_auc:.2f} ± {std_auc:.2f}",
            lw=2)

    #Plot first std as shade
    ax.fill_between(mean_fpr,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    alpha=0.2)

    #Chance line to provide context
    ax.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(model_name)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

plt.suptitle("Mean ROC Curves per Classifier", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#Show best classifier
best_model_name = df_summary.loc[df_summary["Mean AUC"].idxmax(), "Model"]
best_model_row = df_summary.loc[df_summary["Model"] == best_model_name]

mean_acc = best_model_row["Mean Accuracy"].values[0]

print(f"\nBest model based on AUC: {best_model_name}")
print(f"Mean Accuracy: {mean_acc:.3f}")

#Check if accuracy requirement is met for best model (≥ 0.70) to ensure clinical relevance
if mean_acc >= 0.70:
    print("Accuracy requirement met (≥ 0.70)")
else:
    print("Accuracy requirement NOT met (< 0.70)")

#Get pipeline and hyperparameter grid of best performing classifier
best_pipeline, best_param_grid = models[best_model_name]

#Hyperparameter search for final model based on 100% of the data
final_search = RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=best_param_grid,
    n_iter=40,
    cv=inner_cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    refit=True
)

final_search.fit(X, y)

#Train final model with found hyperparameters
final_model = final_search.best_estimator_

#Print final selected hyperparameters
print("\nFinal selected hyperparameters:")
for k, v in final_search.best_params_.items():
    print(f"{k}: {v}")

#Print that final model has been trained
print(f"\nBest mean CV AUC during final tuning: {final_search.best_score_:.4f}")
print(f"Final {best_model_name} model trained on all data.")

