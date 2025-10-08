"""
main.py

A beginner-friendly end-to-end ML pipeline using the Breast Cancer Wisconsin dataset.

Run: python main.py

This script is heavily commented so you can understand every line.
"""

# 1) Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import os
from utils import show_basic_info, evaluate_model, top_features_from_coefs, permutation_importances

# 2) Load dataset
# scikit-learn provides several built-in datasets. load_breast_cancer returns a dictionary-like object.
data = load_breast_cancer()
X = data.data               # feature matrix (numpy array)
y = data.target             # target array (0 = malignant, 1 = benign)
feature_names = data.feature_names

# 3) Convert to pandas DataFrame for easier exploration
df = pd.DataFrame(X, columns=feature_names)

# 4) Basic EDA (exploratory data analysis)
print("=== EXPLORATORY DATA ANALYSIS ===")
show_basic_info(df, y)

# 5) Train-test split
# We split the data into training and test sets so that we evaluate on data the model hasn't seen.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 6) Feature scaling
# Many models (logistic regression, MLP) perform better when features are standardized.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled = scaler.transform(X_test)         # only transform test

# 7) Train models
# a) Logistic Regression (simple, interpretable)
logreg = LogisticRegression(max_iter=10000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# b) Random Forest (non-linear, good baseline)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # tree-based models don't require scaling

# c) MLP (a small neural network)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# 8) Evaluate models
print("=== EVALUATION ON TEST SET ===")
# Logistic Regression
y_pred_log = logreg.predict(X_test_scaled)
y_prob_log = logreg.predict_proba(X_test_scaled)[:,1]
evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)

# Random Forest
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# MLP
y_pred_mlp = mlp.predict(X_test_scaled)
y_prob_mlp = mlp.predict_proba(X_test_scaled)[:,1]
evaluate_model("MLP Classifier", y_test, y_pred_mlp, y_prob_mlp)

# 9) Interpretation / Feature importance
print("=== MODEL INTERPRETATION ===")

# For logistic regression: coefficients tell us how each standardized feature affects the log-odds.
top_pos, top_neg = top_features_from_coefs(feature_names, logreg.coef_, top_n=5)
print("Top positive coefficients (increase odds of class 1):")
print(top_pos)
print("\nTop negative coefficients (decrease odds of class 1):")
print(top_neg)
print()

# For random forest: use built-in feature_importances_
import pandas as pd
rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Top Random Forest importances:")
print(rf_importances.head(10))
print()

# For MLP: use permutation importance to approximate importance
print("Permutation importances for MLP (approx):")
mlp_perms = permutation_importances(mlp, X_test_scaled, y_test, metric=roc_auc_score, n_repeats=20)
mlp_perm_series = pd.Series(mlp_perms, index=feature_names).sort_values(ascending=False)
print(mlp_perm_series.head(10))
print()

# 10) Save a small summary to disk
os.makedirs("models", exist_ok=True)
with open("models/summary.txt", "w") as f:
    f.write("Models trained: LogisticRegression, RandomForest, MLP\n")
    f.write("See console output for evaluation and interpretation.\n")

print("Done. Check the console output for results and `models/summary.txt` for a summary.")