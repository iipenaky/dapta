from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dapta.dae.state_builder import PatientProfile, N_METRICS_PER_TASK, N_TASKS
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

NON_APHASIA_SUBTYPES: Set[str] = {
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
}

# Controls get assigned this reserved cluster ID so they can still be
# tracked / evaluated but never influence the aphasia cluster centroids.
CONTROL_CLUSTER_ID = -1


class PatientClusterer:

    def __init__(self, max_k: int = 10, random_seed: int = 42):
        self.max_k       = max_k
        self.random_seed = random_seed
        self._kmeans:    Optional[KMeans] = None
        self._encoder    = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._scaler     = StandardScaler()
        self.n_clusters: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
        min_k:         int = 6,
    ) -> np.ndarray:
        """
        Cluster aphasia patients only, then assign controls CONTROL_CLUSTER_ID.

        Returns an integer label array of length len(profiles).
        Controls receive label -1 and are excluded from RL cluster agents.
        """
        aphasia_mask = self._aphasia_mask(profiles)
        n_aphasia    = int(aphasia_mask.sum())
        n_controls   = int((~aphasia_mask).sum())

        logger.info(
            f"Clustering {n_aphasia} aphasia patients "
            f"(excluding {n_controls} non-aphasic patients)."
        )

        if n_aphasia == 0:
            raise ValueError(
                "No aphasia patients found. Check NON_APHASIA_SUBTYPES matches "
                "your data's subtype labels."
            )

        # Subset to aphasia patients only
        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )

        # Build feature matrix and fit KMeans
        X = self._fit_transform(aphasia_profiles, aphasia_states)

        # Enforce sensible k bounds
        max_possible = min(self.max_k, n_aphasia - 1)
        min_k        = min(min_k, max_possible)
        self.n_clusters = self.find_elbow_k(X, min_k=min_k, max_k=max_possible)
        logger.info(f"Optimal clusters found: k={self.n_clusters}")

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,
        )
        aphasia_labels = self._kmeans.fit_predict(X)
        logger.info(
            f"Clustered {n_aphasia} aphasia patients into {self.n_clusters} clusters. "
            f"Counts: {np.bincount(aphasia_labels)}"
        )

        # Build full label array — controls get CONTROL_CLUSTER_ID
        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)
        full_labels[aphasia_mask] = aphasia_labels

        return full_labels

    def predict(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Assign new patients to existing clusters.
        Controls are assigned CONTROL_CLUSTER_ID without touching the model.
        """
        if self._kmeans is None:
            raise RuntimeError("Clusterer must be fitted before calling predict().")

        aphasia_mask     = self._aphasia_mask(profiles)
        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )

        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)

        if aphasia_profiles:
            X = self._transform(aphasia_profiles, aphasia_states)
            full_labels[aphasia_mask] = self._kmeans.predict(X)

        return full_labels

    def get_cluster_groups(
        self,
        profiles: List[PatientProfile],
        labels:   np.ndarray,
    ) -> Dict[int, List[PatientProfile]]:
        """
        Returns a dict mapping cluster_id -> list of profiles.
        Controls (label == CONTROL_CLUSTER_ID) are included under key -1
        so callers can inspect them but skip training agents for that key.
        """
        if self.n_clusters is None:
            raise RuntimeError("Clusterer must be fitted first.")

        unique = sorted(set(int(l) for l in labels))
        groups: Dict[int, List[PatientProfile]] = {i: [] for i in unique}
        for profile, label in zip(profiles, labels):
            groups[int(label)].append(profile)
        return groups

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _fit_transform(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        subtypes        = np.array([[p.aphasia_subtype] for p in profiles])
        wab_aq          = np.array([p.wab_aq for p in profiles]).reshape(-1, 1)
        subtype_encoded = self._encoder.fit_transform(subtypes)

        numeric = [wab_aq]
        if state_vectors is not None:
            numeric.append(self._extract_discourse_means(state_vectors))

        numeric_scaled = self._scaler.fit_transform(np.hstack(numeric))
        return np.hstack([subtype_encoded, numeric_scaled]).astype(np.float32)

    def _transform(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        subtypes        = np.array([[p.aphasia_subtype] for p in profiles])
        wab_aq          = np.array([p.wab_aq for p in profiles]).reshape(-1, 1)
        subtype_encoded = self._encoder.transform(subtypes)

        numeric = [wab_aq]
        if state_vectors is not None:
            numeric.append(self._extract_discourse_means(state_vectors))

        numeric_scaled = self._scaler.transform(np.hstack(numeric))
        return np.hstack([subtype_encoded, numeric_scaled]).astype(np.float32)

    # ------------------------------------------------------------------
    # Elbow method
    # ------------------------------------------------------------------

    def compute_inertias(self, X: np.ndarray, max_k: int) -> List[float]:
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_seed,
                n_init=10,
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        return inertias

    def find_elbow_k(self, X: np.ndarray, min_k: int = 6, max_k: int = 10) -> int:
        if max_k < 3:
            return min_k

        inertias      = self.compute_inertias(X, max_k)
        logger.info(f"Inertia values: {inertias}")

        deltas        = np.diff(inertias)
        second_deltas = np.diff(deltas)

        if len(second_deltas) == 0:
            return min_k

        elbow  = int(np.argmax(np.abs(second_deltas))) + 2
        result = max(min_k, min(elbow, max_k))
        logger.info(f"Elbow at k={elbow}, enforcing min_k={min_k} max_k={max_k}, using k={result}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aphasia_mask(profiles: List[PatientProfile]) -> np.ndarray:
        return np.array(
            [p.aphasia_subtype not in NON_APHASIA_SUBTYPES for p in profiles],
            dtype=bool,
        )

    @staticmethod
    def _extract_discourse_means(state_vectors: np.ndarray) -> np.ndarray:
        """
        Returns shape (N, 5) — mean per discourse metric across all tasks.
        """
        N          = state_vectors.shape[0]
        n_metrics  = N_METRICS_PER_TASK   # 5
        n_tasks    = N_TASKS              # 5
        disc_means = np.zeros((N, n_metrics), dtype=np.float32)

        for m in range(n_metrics):
            dims = [t * n_metrics + m for t in range(n_tasks)]
            disc_means[:, m] = state_vectors[:, dims].mean(axis=1)

        return disc_means