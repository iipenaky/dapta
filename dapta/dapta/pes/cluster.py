"""
Patient clustering for personalised RL training.

Groups patients by aphasia subtype and severity to enable
patient-specific policy training (addressing RQ4).
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from dapta.dae.state_builder import PatientProfile, APHASIA_SUBTYPES
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class PatientClusterer:
    """
    Clusters patients using aphasia subtype, WAB-AQ severity, and months post-onset.

    Default: 6 clusters (one per major aphasia subtype).
    Clusters are used to train separate RL policies.

    Parameters
    ----------
    n_clusters  : Number of clusters (default 6)
    random_seed : For reproducibility
    """

    def __init__(self, n_clusters: int = 6, random_seed: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self._kmeans: Optional[KMeans] = None
        self._le = LabelEncoder().fit(APHASIA_SUBTYPES)

    def fit_predict(
        self, profiles: List[PatientProfile]
    ) -> np.ndarray:
        """
        Fit the clusterer and return cluster labels.

        Returns
        -------
        np.ndarray of int, shape (N,)
        """
        X = self._build_feature_matrix(profiles)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,
        )
        labels = self._kmeans.fit_predict(X)
        logger.info(
            f"Clustered {len(profiles)} patients into {self.n_clusters} clusters. "
            f"Counts: {np.bincount(labels)}"
        )
        return labels

    def predict(self, profiles: List[PatientProfile]) -> np.ndarray:
        """Predict cluster for new patients."""
        if self._kmeans is None:
            raise RuntimeError("Clusterer not fitted. Call fit_predict() first.")
        X = self._build_feature_matrix(profiles)
        return self._kmeans.predict(X)

    def get_cluster_groups(
        self,
        profiles: List[PatientProfile],
        labels: np.ndarray,
    ) -> Dict[int, List[PatientProfile]]:
        """
        Returns
        -------
        {cluster_id: [profiles in that cluster]}
        """
        groups: Dict[int, List[PatientProfile]] = {i: [] for i in range(self.n_clusters)}
        for profile, label in zip(profiles, labels):
            groups[int(label)].append(profile)
        return groups

    def _build_feature_matrix(self, profiles: List[PatientProfile]) -> np.ndarray:
        """Build clustering feature matrix: [subtype_encoded, wab_aq_norm, months_norm]."""
        subtypes_enc = self._le.transform([
            p.aphasia_subtype if p.aphasia_subtype in self._le.classes_ else "Other"
            for p in profiles
        ])
        wab_aq = np.array([p.wab_aq for p in profiles]) / 100.0
        months = np.array([min(p.months_post_onset, 60) for p in profiles]) / 60.0

        return np.column_stack([subtypes_enc, wab_aq, months]).astype(np.float32)
