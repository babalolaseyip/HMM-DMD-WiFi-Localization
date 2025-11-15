"""Tests for localization models."""
import pytest
import numpy as np
from models.hmm_dmd import HMMDMDLocalizer


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n_samples = 100
    n_timestamps = 20
    n_aps = 14
    X = -90 + 60 * np.random.rand(n_samples, n_timestamps, n_aps)
    y = 50 * np.random.rand(n_samples, 2)
    rooms = np.random.randint(0, 3, size=n_samples)
    return X, y, rooms


def test_hmm_dmd_init():
    model = HMMDMDLocalizer(dmd_rank=7, n_components=14)
    assert model.dmd_rank == 7
    assert model.n_components == 14


def test_hmm_dmd_fit_predict(synthetic_data):
    X, y, rooms = synthetic_data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    rooms_train = rooms[:split]
    model = HMMDMDLocalizer(dmd_rank=7, n_components=5)
    model.fit(X_train, y_train, rooms_train)
    predictions = model.predict(X_test)
    assert predictions.shape == (len(X_test), 2)
    assert not np.any(np.isnan(predictions))


def test_dmd_feature_extraction(synthetic_data):
    X, _, _ = synthetic_data
    model = HMMDMDLocalizer(dmd_rank=7)
    features = model._extract_dmd_features(X[:10])
    assert features.shape[0] == 10
    assert features.shape[1] == 7
    assert not np.any(np.isnan(features))
