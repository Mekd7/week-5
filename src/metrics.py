from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix
)

class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        return {
            "AUC_PR": average_precision_score(y_test, y_proba),
            "F1": f1_score(y_test, y_pred),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        }
