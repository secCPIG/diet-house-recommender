"""A k nearest neighbors recommender built with scikit-learn.

Idea: to recommend something to you, find the people in the training data who
are most similar to you (age, weight, budget, ...) and see what fit them best.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNRecommender:
    """Learns from examples, then ranks every option for a new person."""

    def __init__(self, feature_columns: list[str], target_column: str, neighbors: int = 15):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.pipeline = Pipeline([
            # Scaling matters: without it, budget (in the thousands) would
            # outweigh age (around 20 to 60) just because the numbers are bigger.
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=neighbors, weights="distance")),
        ])
        self.is_trained = False

    def _check_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

    def train(self, data: pd.DataFrame) -> None:
        """Fit the model on a table of examples."""
        self._check_columns(data, self.feature_columns + [self.target_column])
        self.pipeline.fit(data[self.feature_columns], data[self.target_column])
        self.is_trained = True

    def evaluate(self, data: pd.DataFrame, k: int = 3) -> float:
        """Train on 80% of the data, test on the other 20%.

        Returns top k accuracy: how often the right answer is in the model's top k picks.
        """
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        self.train(train)
        probabilities = self.pipeline.predict_proba(test[self.feature_columns])
        return top_k_accuracy_score(
            test[self.target_column], probabilities, k=k, labels=self.pipeline.classes_
        )

    def rank(self, person: dict) -> list[tuple[str, float]]:
        """Every option the model thinks fits, best first, with a 0 to 1 confidence."""
        if not self.is_trained:
            raise RuntimeError("Call train() before rank().")

        row = pd.DataFrame([person])
        self._check_columns(row, self.feature_columns)
        probabilities = self.pipeline.predict_proba(row[self.feature_columns])[0]

        ranked = [
            (str(label), float(p))
            for label, p in zip(self.pipeline.classes_, probabilities)
            if p > 0
        ]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked