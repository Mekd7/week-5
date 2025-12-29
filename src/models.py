from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:

    @staticmethod
    def logistic_regression():
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    @staticmethod
    def random_forest(n_estimators=300, max_depth=10):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
