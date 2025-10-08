
# Breast Cancer Classifier — Beginner-Friendly ML Project

**Goal:** Build a simple, well-explained machine learning pipeline for binary classification using the built-in Breast Cancer Wisconsin dataset from scikit-learn.  
This project is designed so you can understand every line of code, explain it in interviews, and run it locally.

## What you'll learn
- Loading and exploring a dataset with pandas
- Preprocessing (train/test split, scaling)
- Training three models:
  - Logistic Regression (interpretable linear model)
  - Random Forest (tree-based ensemble)
  - MLP (simple neural network using sklearn's MLPClassifier)
- Evaluating models (accuracy, precision, recall, F1, ROC AUC)
- Interpreting models (coefficients, feature importance, permutation importance)
- Saving results and a reproducible script

## Files in this project
- `main.py` — The main script that runs the entire pipeline end-to-end.
- `utils.py` — Helper functions for EDA, evaluation, and interpretation.
- `requirements.txt` — Python packages required.
- `README.md` — This file.
- `explanations.md` — Line-by-line explanation of `main.py` (for interview prep).

## How to run
1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. Inspect printed outputs and the saved `models/` folder (if created).

## Notes for interviews
- Be ready to explain what each model does, why we scale features for certain models, and how to interpret the results.
- This project focuses on clarity and explanation rather than squeezing maximum performance.

