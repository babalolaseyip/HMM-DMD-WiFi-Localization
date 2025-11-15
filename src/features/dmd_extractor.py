"""Dynamic Mode Decomposition feature extractor."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DMDExtractor(BaseEstimator, TransformerMixin):
    """Extract DMD features from time-series RSSI data."""

    def __init__(self, rank: int = 7, reconstruction_method: str = 'mean_magnitude'):
        self.rank = rank
        self.reconstruction_method = reconstruction_method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.extract_features(X)

    def extract_features(self, X):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, X.shape[1])
        features = []
        for sequence in X:
            try:
                dmd_modes = self._compute_dmd_modes(sequence)
                feature_vector = self._reconstruct_features(dmd_modes)
                features.append(feature_vector)
            except Exception:
                features.append(np.zeros(self.rank))
        return np.array(features)

    def _compute_dmd_modes(self, X):
        if X.shape[0] < 2:
            return np.zeros((X.shape[1], min(self.rank, X.shape[1])))
        X1 = X[:-1].T
        X2 = X[1:].T
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        r = min(self.rank, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vh[:r, :].T
        S_r_inv = np.diag(1.0 / (s_r + 1e-10))
        M_hat = U_r.T @ X2 @ V_r @ S_r_inv
        eigenvalues, eigenvectors = np.linalg.eig(M_hat)
        dmd_modes = X2 @ V_r @ S_r_inv @ eigenvectors
        return dmd_modes

    def _reconstruct_features(self, dmd_modes):
        if self.reconstruction_method == 'mean_magnitude':
            features = np.mean(np.abs(dmd_modes), axis=0)
        elif self.reconstruction_method == 'max_magnitude':
            features = np.max(np.abs(dmd_modes), axis=0)
        elif self.reconstruction_method == 'real_part':
            features = np.mean(np.real(dmd_modes), axis=0)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")
        features = np.asarray(features).real
        if features.shape[0] < self.rank:
            padded = np.zeros(self.rank)
            padded[: features.shape[0]] = features
            return padded
        return features[: self.rank]
