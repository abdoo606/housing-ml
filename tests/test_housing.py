from __future__ import annotations

import numpy as np
import pytest

from housing.pipeline import Encoder, Ridge, cross_validate, kfold_indices, metrics, train

ROWS = [
    {"area": 60, "rooms": 2, "age": 30, "neighbourhood": "north"},
    {"area": 80, "rooms": 3, "age": 20, "neighbourhood": "north"},
    {"area": 100, "rooms": 3, "age": 10, "neighbourhood": "south"},
    {"area": 120, "rooms": 4, "age": 5, "neighbourhood": "south"},
    {"area": 150, "rooms": 4, "age": 2, "neighbourhood": "centre"},
    {"area": 90, "rooms": 2, "age": 40, "neighbourhood": "centre"},
    {"area": 70, "rooms": 2, "age": 25, "neighbourhood": "north"},
    {"area": 110, "rooms": 3, "age": 15, "neighbourhood": "south"},
]
PRICES = [200, 260, 320, 380, 460, 280, 225, 350]


def test_encoder_shapes_and_one_hot():
    encoder = Encoder().fit(ROWS)
    X = encoder.transform(ROWS)
    assert X.shape == (8, 3 + 3)  # 3 numeric + one-hot of 3 neighbourhoods
    assert set(np.unique(X[:, 3:])) <= {0.0, 1.0}
    assert np.allclose(X[:, 3:].sum(axis=1), 1.0)


def test_standardisation_uses_train_statistics():
    encoder = Encoder().fit(ROWS)
    X = encoder.transform(ROWS)
    assert abs(X[:, 0].mean()) < 1e-9


def test_unfitted_transform_raises():
    with pytest.raises(RuntimeError):
        Encoder().transform(ROWS)


def test_fit_on_empty_raises():
    with pytest.raises(ValueError):
        Encoder().fit([])


def test_ridge_recovers_a_linear_relationship():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 3 * X[:, 0] + 7
    model = Ridge(alpha=0.0).fit(X, y)
    assert model.weights[0] == pytest.approx(3.0, abs=1e-6)
    assert model.intercept == pytest.approx(7.0, abs=1e-6)


def test_ridge_shrinks_weights():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = X @ np.array([1.0, 2.0, 3.0])
    small = Ridge(alpha=0.001).fit(X, y)
    big = Ridge(alpha=100.0).fit(X, y)
    assert np.linalg.norm(big.weights) < np.linalg.norm(small.weights)


def test_ridge_validation():
    with pytest.raises(ValueError):
        Ridge().fit(np.zeros((2, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        Ridge().fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(RuntimeError):
        Ridge().predict(np.zeros((1, 1)))


def test_metrics_are_correct():
    result = metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert result["mae"] == 1.0
    assert result["rmse"] == pytest.approx(1.414, abs=0.01)


def test_metrics_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics(y, y)["r2"] == 1.0


def test_kfold_partitions_every_row_exactly_once():
    folds = kfold_indices(10, 5, seed=1)
    assert sorted(np.concatenate(folds).tolist()) == list(range(10))


def test_kfold_validation():
    with pytest.raises(ValueError):
        kfold_indices(5, 1)
    with pytest.raises(ValueError):
        kfold_indices(5, 9)


def test_cross_validation_returns_means():
    result = cross_validate(ROWS, PRICES, k=4, alpha=0.5)
    assert result["folds"] == 4
    assert result["mae"] > 0
    assert result["r2"] <= 1


def test_model_learns_the_direction_of_the_price():
    model = train(ROWS, PRICES)
    cheap = model.predict([{"area": 50, "rooms": 2, "age": 30, "neighbourhood": "north"}])[0]
    pricey = model.predict([{"area": 160, "rooms": 4, "age": 2, "neighbourhood": "centre"}])[0]
    assert pricey > cheap


def test_unseen_category_is_encoded_as_all_zeros():
    model = train(ROWS, PRICES)
    prediction = model.predict([{"area": 100, "rooms": 3, "age": 10, "neighbourhood": "unknown"}])
    assert prediction.shape == (1,)
