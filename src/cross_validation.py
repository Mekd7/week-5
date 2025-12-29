import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score

class CrossValidator:
    def __init__(self, model, k=5):
        self.model = model
        self.k = k

    def run(self, X, y):
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        aucs, f1s = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model.fit(X_train, y_train)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            y_pred = self.model.predict(X_val)

            aucs.append(average_precision_score(y_val, y_proba))
            f1s.append(f1_score(y_val, y_pred))

        return {
            "AUC_PR_mean": np.mean(aucs),
            "AUC_PR_std": np.std(aucs),
            "F1_mean": np.mean(f1s),
            "F1_std": np.std(f1s)
        }
