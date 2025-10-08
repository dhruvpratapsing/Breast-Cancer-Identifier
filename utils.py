"""
utils.py

Helper functions used by main.py:
- show_basic_info: prints dataset overview
- evaluate_model: prints evaluation metrics
- plot_feature_importance: (optional) creates a simple bar plot for feature importances
- permutation_importances: compute permutation importance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def show_basic_info(df, target):
    print("Shape of dataset:", df.shape)
    print("Features:", df.columns.tolist())
    print("Target distribution:")
    print(pd.Series(target).value_counts())
    print("\nSample rows:\n", df.head().to_string())

def evaluate_model(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"--- {name} Evaluation ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"ROC AUC  : {auc:.4f}")
        except Exception as e:
            print("ROC AUC could not be computed:", e)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()

def top_features_from_coefs(feature_names, coefs, top_n=10):
    # For linear models: show top positive and negative coefficients
    coef_series = pd.Series(coefs.flatten(), index=feature_names)
    top_pos = coef_series.sort_values(ascending=False).head(top_n)
    top_neg = coef_series.sort_values().head(top_n)
    return top_pos, top_neg

def permutation_importances(model, X, y, metric, n_repeats=10, random_state=42):
    """
    Simple permutation importance implementation using sklearn's permutation_importance if available.
    Falls back to naive implementation otherwise.
    """
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, scoring=metric, n_repeats=n_repeats, random_state=random_state)
        importances = r.importances_mean
        return importances
    except Exception:
        # Naive fallback (slower): shuffle each column and measure metric drop
        import numpy as np
        baseline = metric(y, model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else model.predict(X))
        importances = []
        X_copy = X.copy()
        rng = np.random.RandomState(random_state)
        for i in range(X.shape[1]):
            saved = X_copy[:, i].copy()
            scores = []
            for _ in range(n_repeats):
                rng.shuffle(X_copy[:, i])
                val = metric(y, model.predict_proba(X_copy)[:,1] if hasattr(model, "predict_proba") else model.predict(X_copy))
                scores.append(baseline - val)
            X_copy[:, i] = saved
            importances.append(np.mean(scores))
        return np.array(importances)