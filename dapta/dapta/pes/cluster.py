"""
Patient clustering for personalised RL training.

Clusters patients based on:
- Aphasia subtype
- WAB-AQ severity
- Months post-onset

The optimal number of clusters is determined automatically
using the K-Means inertia elbow method.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dapta.dae.state_builder import PatientProfile
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class PatientClusterer:
    """
    Patient clustering using K-Means with automatic cluster selection
    via inertia (elbow method).
    """

    def __init__(self, max_k: int = 10, random_seed: int = 42):
        self.max_k = max_k
        self.random_seed = random_seed
        self._kmeans: Optional[KMeans] = None
        self._encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self._scaler = StandardScaler()
        self.n_clusters: Optional[int] = None

    def fit_predict(self, profiles: List[PatientProfile]) -> np.ndarray:
        """
        Automatically determine best k using inertia
        then run clustering.
        """

        X = self.build_feature_matrix(profiles)

        # determine best k
        self.n_clusters = self.find_elbow_k(X)

        logger.info(f"Optimal clusters found: k={self.n_clusters}")

        self._kmeans = KMeans(n_clusters=self.n_clusters,random_state=self.random_seed,n_init=10)

        labels = self._kmeans.fit_predict(X)

        logger.info(
            f"Clustered {len(profiles)} patients into {self.n_clusters} clusters. "
            f"Counts: {np.bincount(labels)}"
        )

        return labels

    def predict(self, profiles):
        """Predict cluster labels for new patients."""
        if self._kmeans is None:
            raise RuntimeError("Clusterer must be fitted first.")

        X = self.build_feature_matrix(profiles)
        return self._kmeans.predict(X)

    def get_cluster_groups(self,profiles: List[PatientProfile],labels: np.ndarray):
        """
        Return dictionary mapping cluster_id -> patient profiles.
        """

        groups: Dict[int, List[PatientProfile]] = {
            i: [] for i in range(self.n_clusters)
        }

        for profile, label in zip(profiles, labels):
            groups[int(label)].append(profile)

        return groups

    def build_feature_matrix(self, profiles):
        """
        Build clustering feature matrix:
        [aphasia subtype (one-hot), wab_aq, months_post_onset]
        """

        subtypes = np.array([[p.aphasia_subtype] for p in profiles])

        wab_aq = np.array([p.wab_aq for p in profiles]).reshape(-1, 1)

        months = np.array([
            min(p.months_post_onset, 60) for p in profiles
        ]).reshape(-1, 1)

        numeric_features = np.hstack([wab_aq, months])

        numeric_scaled = self._scaler.fit_transform(numeric_features)

        subtype_encoded = self._encoder.fit_transform(subtypes)

        X = np.hstack([subtype_encoded, numeric_scaled])

        return X.astype(np.float32)

    def compute_inertias(self, X):
        """
        Compute inertia values for k=1..max_k
        """

        inertias = []

        for k in range(1, self.max_k + 1):

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_seed,
                n_init=10,
            )

            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        return inertias

    def find_elbow_k(self, X):
        """
        Detect elbow point from inertia curve using second derivative.
        """

        inertias = self.compute_inertias(X)

        logger.info(f"Inertia values: {inertias}")

        # compute curvature
        deltas = np.diff(inertias)
        second_deltas = np.diff(deltas)

        elbow = np.argmax(np.abs(second_deltas)) + 2

        return max(2, elbow)