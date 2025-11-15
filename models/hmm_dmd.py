"""HMM-DMD Indoor Localization Model."""
import numpy as np
try:
    from hmmlearn import hmm
except Exception:
    hmm = None
from sklearn.cluster import KMeans
from .base_model import BaseLocalizationModel


class HMMDMDLocalizer(BaseLocalizationModel):
    """WiFi Fingerprinting using HMM and DMD feature extraction.

    Implementation of the method described in:
    Babalola & Balyan (2021) - WiFi Fingerprinting Indoor Localization
    Based on Dynamic Mode Decomposition Feature Selection with HMM.

    Args:
        dmd_rank: Rank for DMD decomposition (default: 7)
        n_components: Number of HMM states (default: 14)
        n_mix: Number of Gaussian mixtures (default: 3)
        cov_type: Covariance type for GMM-HMM (default: 'diag')
    """

    def __init__(self, dmd_rank=7, n_components=14, n_mix=3, cov_type='diag'):
        self.dmd_rank = dmd_rank
        self.n_components = n_components
        self.n_mix = n_mix
        self.cov_type = cov_type
        self.hmm_models = {}
        self.state_coordinates = {}

    def fit(self, X, y, rooms=None):
        """Train HMM models for each room.

        Args:
            X: RSSI measurements (n_samples, n_timestamps, n_aps)
            y: Ground truth coordinates (n_samples, 2)
            rooms: Room labels (n_samples,)
        """
        feature_vectors = self._extract_dmd_features(X)

        if rooms is None:
            rooms = np.zeros(len(X))

        for room_idx, room in enumerate(np.unique(rooms)):
            room_mask = rooms == room
            room_features = feature_vectors[room_mask]
            room_coords = y[room_mask]
            hmm_model = self._initialize_hmm(room_features, room_coords, room_idx)
            self.hmm_models[room] = hmm_model

        return self

    def predict(self, X):
        """Predict locations using trained HMM models."""
        feature_vectors = self._extract_dmd_features(X)
        predictions = []
        for features in feature_vectors:
            best_room, best_state = self._viterbi_decode(features)
            coords = self.state_coordinates.get(best_room, {}).get(best_state, np.array([0.0, 0.0]))
            predictions.append(coords)
        return np.array(predictions)

    def _extract_dmd_features(self, X):
        """Extract DMD features from RSSI sequences."""
        X = np.asarray(X)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, X.shape[1])
        features = []
        for sequence in X:
            dmd_modes = self._compute_dmd_modes(sequence)
            feature_vector = self._reconstruct_features(dmd_modes)
            features.append(feature_vector)
        return np.array(features)

    def _compute_dmd_modes(self, X):
        """Compute DMD modes following paper equations."""
        if X.shape[0] < 2:
            return np.zeros((X.shape[1], min(self.dmd_rank, X.shape[1])))
        X1 = X[:-1].T
        X2 = X[1:].T
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
        r = min(self.dmd_rank, len(s))
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vh[:r, :].T
        S_r_inv = np.diag(1.0 / (s_r + 1e-10))
        M_hat = U_r.T @ X2 @ V_r @ S_r_inv
        eigenvalues, eigenvectors = np.linalg.eig(M_hat)
        dmd_modes = X2 @ V_r @ S_r_inv @ eigenvectors
        return dmd_modes

    def _reconstruct_features(self, dmd_modes):
        features = np.mean(np.abs(dmd_modes), axis=0)
        if features.shape[0] < self.dmd_rank:
            padded = np.zeros(self.dmd_rank)
            padded[: features.shape[0]] = features
            return padded
        return features[: self.dmd_rank]

    def _initialize_hmm(self, features, coordinates, room_id):
        n_components = min(self.n_components, max(1, len(features)))
        kmeans = KMeans(n_clusters=n_components, random_state=42)
        state_labels = kmeans.fit_predict(features)
        self.state_coordinates[room_id] = {}
        for state in range(n_components):
            state_mask = state_labels == state
            if np.any(state_mask):
                mean_coords = np.mean(coordinates[state_mask], axis=0)
                self.state_coordinates[room_id][state] = mean_coords
            else:
                self.state_coordinates[room_id][state] = np.mean(coordinates, axis=0)

        if hmm is None:
            class _DummyModel:
                def score(self, X):
                    return -np.sum(np.var(X, axis=0))
                def predict(self, X):
                    return np.zeros(X.shape[0], dtype=int)
            return _DummyModel()

        model = hmm.GMMHMM(
            n_components=n_components,
            n_mix=self.n_mix,
            covariance_type=self.cov_type,
            n_iter=100,
            random_state=42,
        )
        try:
            lengths = [len(features)]
            model.fit(features, lengths)
        except Exception:
            try:
                model.fit(features)
            except Exception:
                pass
        return model

    def _viterbi_decode(self, features):
        best_score = -float('inf')
        best_room = None
        best_path = None
        features = np.asarray(features)
        features_seq = features.reshape(-1, features.shape[-1])
        for room, hmm_model in self.hmm_models.items():
            try:
                score = hmm_model.score(features_seq)
                if score > best_score:
                    best_score = score
                    best_room = room
                    best_path = hmm_model.predict(features_seq)
            except Exception:
                continue
        if best_path is not None and len(best_path) > 0:
            best_state = int(np.bincount(best_path).argmax())
        else:
            best_state = 0
        return best_room, best_state
