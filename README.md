# Machine Learning Project – Lipoma vs WDLPS Classification

This repository contains a machine learning project developed for the Technical Medicine Master’s program.

## Project Description
In this study, we develop and evaluate machine learning models to classify lipoma and well-differentiated liposarcoma (WDLPS). The models use features extracted from T1-weighted MRI scans from the WORC (Workflow for Optimal Radiomics Classification) dataset.

## Repository Structure
The repository consists of several Python files:

* **final_main.py**
  This is the main file of the project. It combines all steps and runs the full pipeline.

* **preprocessing.py**
  Contains all preprocessing steps applied to the data before training the models.

* **Classifier files:**

  * **svm.py** – Support Vector Machine classifier
  * **rf.py** – Random Forest classifier
  * **xgboost.py** – XGBoost classifier


## How to Run
To run the full project, simply execute:
final_main.py

This will:
1. Load and preprocess the data
2. Train the models
3. Evaluate their performance

## Notes
* Make sure all required Python packages are installed (e.g., scikit-learn, xgboost, numpy, pandas).
* Results may vary slightly depending on the Python version and the versions of the installed packages.

## Authors
Technical Medicine Students, Group 11
Wouter van Wijck (5527562), Sarah van der Hart (5743931), Tygo Hillen (6236227) and Lara Verhoef (5297788)
