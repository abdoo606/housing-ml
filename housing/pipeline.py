"""Feature engineering + ridge regression + honest cross-validation, in NumPy.

No scikit-learn: the point is to show the maths, the leakage risks and the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Encoder:
    """One-hot for categoricals and standardisation for numerics, fitted on train only."""

    numeric: list[str] = field(default_factory=lambda: ["area", "rooms", "age"])
    categorical: list[str] = field(default_factory=lambda: ["neighbourhood"])
    categories: dict[str, list[str]] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: list[dict]) -> "Encoder":
        if not rows:
            raise ValueError("cannot fit on zero rows")
        for column in self.categorical:
            self.categories[column] = sorted({str(row[column]) for row in rows})
        for column in self.numeric:
            values = np.array([float(row[column]) for row in rows])
            self.means[column] = float(values.mean())
            self.stds[column] = float(values.std()) or 1.0
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        if not self.categories:
            raise RuntimeError("encoder is not fitted")
        features = []
        for row in rows:
            vector = [(float(row[c]) - self.means[c]) / self.stds[c] for c in self.numeric]
            for column in self.categorical:
                values = self.categories[column]
                one_hot = [0.0] * len(values)
                value = str(row[column])
                if value in values:
                    one_hot[values.index(value)] = 1.0
                vector.extend(one_hot)
            features.append(vector)
        return np.array(features, dtype=float)

    def fit_transform(self, rows: list[dict]) -> np.ndarray:
        return self.fit(rows).transform(rows)


@dataclass(slots=True)
class Ridge:
    """Closed-form ridge: w = (X^T X + alpha I)^-1 X^T y, with an unpenalised intercept."""

    alpha: float = 1.0
    weights: np.ndarray | None = None
    intercept: float = 0.0
    feature_means: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y disagree on row count")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on zero rows")
        # Centring both X and y is what makes the intercept free of the penalty.
        self.feature_means = X.mean(axis=0)
        Xc = X - self.feature_means
        yc = y - y.mean()
        identity = np.eye(X.shape[1])
        self.weights = np.linalg.solve(Xc.T @ Xc + self.alpha * identity, Xc.T @ yc)
        self.intercept = float(y.mean() - self.feature_means @ self.weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return X @ self.weights + self.intercept


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2)) or 1.0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(1 - ss_res / ss_tot, 3)}


def kfold_indices(n: int, k: int, *, seed: int = 0) -> list[np.ndarray]:
    if k < 2 or k > n:
        raise ValueError("k must be between 2 and the number of rows")
    order = np.random.default_rng(seed).permutation(n)
    return [fold for fold in np.array_split(order, k)]


def cross_validate(
    rows: list[dict], targets: list[float], *, k: int = 5, alpha: float = 1.0, seed: int = 0
) -> dict[str, object]:
    """Encoder and model are refitted inside every fold - no leakage across the boundary."""
    y = np.array(targets, dtype=float)
    folds = kfold_indices(len(rows), k, seed=seed)
    scores = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        encoder = Encoder().fit([rows[j] for j in train_idx])
        model = Ridge(alpha=alpha).fit(encoder.transform([rows[j] for j in train_idx]), y[train_idx])
        predicted = model.predict(encoder.transform([rows[j] for j in test_idx]))
        scores.append(metrics(y[test_idx], predicted))
    return {
        "folds": len(scores),
        "mae": round(float(np.mean([s["mae"] for s in scores])), 2),
        "rmse": round(float(np.mean([s["rmse"] for s in scores])), 2),
        "r2": round(float(np.mean([s["r2"] for s in scores])), 3),
    }


@dataclass(slots=True)
class TrainedModel:
    encoder: Encoder
    ridge: Ridge

    def predict(self, rows: list[dict]) -> np.ndarray:
        return self.ridge.predict(self.encoder.transform(rows))


def train(rows: list[dict], targets: list[float], *, alpha: float = 1.0) -> TrainedModel:
    encoder = Encoder().fit(rows)
    model = Ridge(alpha=alpha).fit(encoder.transform(rows), np.array(targets, dtype=float))
    return TrainedModel(encoder, model)
