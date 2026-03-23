<<<<<<< Updated upstream
"""
Patient clustering for personalised RL training.

Groups patients by aphasia subtype and severity to enable
patient-specific policy training (addressing RQ4).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
=======
from typing import Dict, List, Optional, Set, Union
>>>>>>> Stashed changes

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from dapta.dae.state_builder import PatientProfile, APHASIA_SUBTYPES
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

<<<<<<< Updated upstream
=======
NON_APHASIA_SUBTYPES: Set[str] = {
    # Raw labels from AphasiaBank CHAT files (before _normalise_subtype)
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
    # Normalised fallback — in case state_builder ever maps these to Other
    # We do NOT include "Other" here because Other is a valid aphasia subtype
}

CONTROL_CLUSTER_ID = -1

>>>>>>> Stashed changes

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

    def __init__(self, n_clusters: int = 8, random_seed: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self._kmeans: Optional[KMeans] = None
        self._le = LabelEncoder().fit(APHASIA_SUBTYPES)

    def fit_predict(
<<<<<<< Updated upstream
        self, profiles: List[PatientProfile]
    ) -> np.ndarray:
        """
        Fit the clusterer and return cluster labels.
=======
        self,
        profiles:        List[PatientProfile],
        state_vectors:   Optional[np.ndarray] = None,
        min_k:           int = 6,
        raw_subtypes:    Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Cluster aphasia patients and return per-patient cluster labels.

        Parameters
        ----------
        profiles : List[PatientProfile]
            PatientProfile objects (subtypes already normalised by state_builder).
        state_vectors : np.ndarray, optional
            Full 47-dim state vectors aligned with profiles.
        min_k : int
            Minimum number of clusters to enforce.
        raw_subtypes : List[str], optional
            Raw subtype strings from patient_profiles.json, aligned with
            profiles.  When provided, these are used for the aphasia mask
            instead of the normalised PatientProfile.aphasia_subtype — this
            correctly excludes controls whose raw label is "NotAphasicByWAB"
            but whose normalised label is "Other".
        """
        aphasia_mask = self._aphasia_mask(profiles, raw_subtypes)
        n_aphasia    = int(aphasia_mask.sum())
        n_controls   = int((~aphasia_mask).sum())

        logger.info(
            f"Clustering {n_aphasia} aphasia patients "
            f"(excluding {n_controls} non-aphasic patients)."
        )

        if n_aphasia == 0:
            raise ValueError(
                "No aphasia patients found. Check NON_APHASIA_SUBTYPES matches "
                "your data's subtype labels, or pass raw_subtypes from "
                "patient_profiles.json."
            )

        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )
        X = self._fit_transform(aphasia_profiles, aphasia_states)

        max_possible    = min(self.max_k, n_aphasia - 1)
        min_k           = min(min_k, max_possible)
        self.n_clusters = self.find_elbow_k(X, min_k=min_k, max_k=max_possible)
        logger.info(f"Optimal clusters found: k={self.n_clusters}")
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    def predict(self, profiles: List[PatientProfile]) -> np.ndarray:
        """Predict cluster for new patients."""
        if self._kmeans is None:
            raise RuntimeError("Clusterer not fitted. Call fit_predict() first.")
        X = self._build_feature_matrix(profiles)
        return self._kmeans.predict(X)
=======
        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)
        full_labels[aphasia_mask] = aphasia_labels

        return full_labels

    def predict(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
        raw_subtypes:  Optional[List[str]]  = None,
    ) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("Clusterer must be fitted before calling predict().")

        aphasia_mask     = self._aphasia_mask(profiles, raw_subtypes)
        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )

        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)

        if aphasia_profiles:
            X = self._transform(aphasia_profiles, aphasia_states)
            full_labels[aphasia_mask] = self._kmeans.predict(X)

        return full_labels
>>>>>>> Stashed changes

    def get_cluster_groups(
        self,
        profiles: List[PatientProfile],
        labels: np.ndarray,
    ) -> Dict[int, List[PatientProfile]]:
<<<<<<< Updated upstream
        """
        Returns
        -------
        {cluster_id: [profiles in that cluster]}
        """
        groups: Dict[int, List[PatientProfile]] = {i: [] for i in range(self.n_clusters)}
=======
        if self.n_clusters is None:
            raise RuntimeError("Clusterer must be fitted first.")

        unique = sorted(set(int(l) for l in labels))
        groups: Dict[int, List[PatientProfile]] = {i: [] for i in unique}
>>>>>>> Stashed changes
        for profile, label in zip(profiles, labels):
            groups[int(label)].append(profile)
        return groups

    def _build_feature_matrix(self, profiles: List[PatientProfile]) -> np.ndarray:
        """Build clustering feature matrix: [subtype_encoded, severity_band, months_norm]."""
        subtypes_enc = self._le.transform([
            p.aphasia_subtype if p.aphasia_subtype in self._le.classes_ else "Other"
            for p in profiles
        ])

        # Discretise WAB-AQ into 4 severity bands using standard clinical cutoffs
        # (Kertesz, 1982): Severe <25, Moderate 25-50, Mild-moderate 50-75, Mild 75+
        def _severity_band(wab_aq: float) -> float:
            if wab_aq < 25:   return 0.0
            elif wab_aq < 50: return 0.33
            elif wab_aq < 75: return 0.67
            else:             return 1.0

        severity = np.array([_severity_band(p.wab_aq) for p in profiles])
        months = np.array([min(p.months_post_onset, 60) for p in profiles]) / 60.0

<<<<<<< Updated upstream
        return np.column_stack([subtypes_enc, severity, months]).astype(np.float32)
=======
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

    def find_elbow_k(
        self, X: np.ndarray, min_k: int = 6, max_k: int = 10
    ) -> int:
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
        logger.info(
            f"Elbow at k={elbow}, enforcing min_k={min_k} "
            f"max_k={max_k}, using k={result}"
        )
        return result

    @staticmethod
    def _aphasia_mask(
        profiles:     List[PatientProfile],
        raw_subtypes: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build a boolean mask: True = aphasia patient, False = control.

        If raw_subtypes is provided, uses those strings for the check —
        this correctly catches controls whose raw label is "NotAphasicByWAB"
        but whose normalised PatientProfile.aphasia_subtype is "Other".

        If raw_subtypes is not provided, falls back to the normalised
        PatientProfile.aphasia_subtype (legacy behaviour).
        """
        if raw_subtypes is not None:
            if len(raw_subtypes) != len(profiles):
                raise ValueError(
                    f"raw_subtypes length ({len(raw_subtypes)}) must match "
                    f"profiles length ({len(profiles)})."
                )
            return np.array(
                [s not in NON_APHASIA_SUBTYPES for s in raw_subtypes],
                dtype=bool,
            )

        # Fallback — normalised subtypes (may miss controls mapped to Other)
        logger.warning(
            "raw_subtypes not provided to _aphasia_mask. "
            "Controls whose subtype was normalised to 'Other' will NOT be "
            "excluded. Pass raw_subtypes from patient_profiles.json for "
            "correct behaviour."
        )
        return np.array(
            [p.aphasia_subtype not in NON_APHASIA_SUBTYPES for p in profiles],
            dtype=bool,
        )

    @staticmethod
    def _extract_discourse_means(state_vectors: np.ndarray) -> np.ndarray:
        N          = state_vectors.shape[0]
        n_metrics  = N_METRICS_PER_TASK
        n_tasks    = N_TASKS
        disc_means = np.zeros((N, n_metrics), dtype=np.float32)

        for m in range(n_metrics):
            dims = [t * n_metrics + m for t in range(n_tasks)]
            disc_means[:, m] = state_vectors[:, dims].mean(axis=1)

        return disc_means
>>>>>>> Stashed changes
