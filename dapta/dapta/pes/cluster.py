from typing import Dict, List, Optional, Set, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dapta.dae.state_builder import PatientProfile, N_METRICS_PER_TASK, N_TASKS
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

# Subtype labels that indicate a patient does not have aphasia.
# These patients are excluded from clustering because clustering is only
# meaningful for patients with aphasia, not neurologically intact controls.
NON_APHASIA_SUBTYPES: Set[str] = {
    # Raw labels from AphasiaBank CHAT files (before _normalise_subtype)
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy"}

# Label assigned to non-aphasia (control) patients in the output array.
# Using -1 distinguishes controls from any valid cluster index (0, 1, 2, ...).
CONTROL_CLUSTER_ID = -1


class PatientClusterer:
    """Groups aphasia patients into clusters based on their profile and speech data.

    Clustering allows the RL system to train a separate agent per group,
    so each agent learns therapy strategies tailored to that patient subtype
    and severity profile rather than averaging across all patients.
    """

    def __init__(self, max_k: int = 10, random_seed: int = 42):
        self.max_k       = max_k
        self.random_seed = random_seed
        # The fitted KMeans model; None until fit_predict has been called.
        self._kmeans:    Optional[KMeans] = None
        # Converts aphasia subtype labels into numbers the model can use.
        self._encoder    = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Puts numeric features on the same scale so no single feature dominates.
        self._scaler     = StandardScaler()
        # Set during fit_predict once the optimal k has been determined.
        self.n_clusters: Optional[int] = None

    def fit_predict(self,profiles:List[PatientProfile],state_vectors:Optional[np.ndarray] = None,min_k:int = 6,raw_subtypes:Optional[List[str]] = None) -> np.ndarray:
        """Trains the clusterer and returns a cluster label for every patient.
        Control patients always receive CONTROL_CLUSTER_ID (-1).

        Parameters
        ----------
        profiles      : one PatientProfile per patient (subtype, WAB-AQ, etc.)
        state_vectors : optional (N, state_dim) array of discourse state vectors;
                        when provided, averaged discourse metrics are included as
                        clustering features alongside subtype and severity.
        min_k         : lower bound on the number of clusters; set to 6 to mirror
                        the six primary WAB aphasia subtypes.
        raw_subtypes  : original subtype strings from patient_profiles.json used
                        to identify controls before subtype normalisation; passing
                        this avoids controls being silently mapped to 'Other'.
        """

        # Build a boolean mask: True for aphasia patients, False for controls.
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

        # Work only with aphasia patients from this point on.
        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        # Slice the state matrix to keep only aphasia-patient rows, if provided.
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )
        # Build and scale the feature matrix for clustering.
        X = self._fit_transform(aphasia_profiles, aphasia_states)

        # Choose a valid k range and find the best number of clusters automatically.
        # Cap max_possible so k is never >= n_aphasia (KMeans requirement).
        max_possible    = min(self.max_k, n_aphasia - 1)
        min_k           = min(min_k, max_possible)
        self.n_clusters = self.find_elbow_k(X, min_k=min_k, max_k=max_possible)
        logger.info(f"Optimal clusters found: k={self.n_clusters}")

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_seed,
            n_init=10,          # Run 10 initialisations to reduce local-optima risk.
        )
        aphasia_labels = self._kmeans.fit_predict(X)
        logger.info(
            f"Clustered {n_aphasia} aphasia patients into {self.n_clusters} clusters. "
            f"Counts: {np.bincount(aphasia_labels)}"
        )

        # Build the final label array for all patients, defaulting controls to -1.
        # This preserves alignment between the label array and the original
        # profiles list so callers can index both with the same integer index.
        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)
        full_labels[aphasia_mask] = aphasia_labels

        return full_labels

    def predict(self,profiles:      List[PatientProfile],state_vectors: Optional[np.ndarray] = None,raw_subtypes:  Optional[List[str]]  = None) -> np.ndarray:
        """Assigns cluster labels to new patients using an already trained clusterer.

        Used at evaluation time to route held-out test patients to the correct
        cluster-specific RL agent without retraining the clusterer.
        """

        if self._kmeans is None:
            raise RuntimeError("Clusterer must be fitted before calling predict().")

        aphasia_mask     = self._aphasia_mask(profiles, raw_subtypes)
        aphasia_profiles = [p for p, m in zip(profiles, aphasia_mask) if m]
        aphasia_states   = (
            state_vectors[aphasia_mask] if state_vectors is not None else None
        )

        # Initialise all labels to -1; only aphasia patients will be updated below.
        full_labels = np.full(len(profiles), CONTROL_CLUSTER_ID, dtype=int)

        if aphasia_profiles:
            # Apply the saved encoder and scaler (no refitting) to new patients.
            X = self._transform(aphasia_profiles, aphasia_states)
            full_labels[aphasia_mask] = self._kmeans.predict(X)

        return full_labels

    def get_cluster_groups(
        self,
        profiles: List[PatientProfile],
        labels:   np.ndarray,
    ) -> Dict[int, List[PatientProfile]]:
        """Returns a dictionary mapping each cluster ID to the patients in that cluster.

        Used to partition training data so each cluster-specific RL agent
        trains only on patients from its own group.
        """

        if self.n_clusters is None:
            raise RuntimeError("Clusterer must be fitted first.")

        unique = sorted(set(int(l) for l in labels))
        groups: Dict[int, List[PatientProfile]] = {i: [] for i in unique}
        for profile, label in zip(profiles, labels):
            groups[int(label)].append(profile)
        return groups

    def _fit_transform(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Builds and scales the feature matrix used to train the clusterer.

        Features are: one-hot encoded aphasia subtype, normalised WAB-AQ score,
        and (optionally) per-metric discourse means averaged across all tasks.
        Combining clinical severity with discourse patterns lets the clusterer
        separate patients by both subtype and functional communication profile.
        """

        subtypes        = np.array([[p.aphasia_subtype] for p in profiles])
        wab_aq          = np.array([p.wab_aq for p in profiles]).reshape(-1, 1)
        # Encode subtype strings as numeric columns (one column per subtype).
        subtype_encoded = self._encoder.fit_transform(subtypes)

        numeric = [wab_aq]
        if state_vectors is not None:
            # Add averaged speech task metrics as extra features.
            numeric.append(self._extract_discourse_means(state_vectors))

        # Fit the scaler on training data only; applied without refitting in _transform.
        numeric_scaled = self._scaler.fit_transform(np.hstack(numeric))
        return np.hstack([subtype_encoded, numeric_scaled]).astype(np.float32)

    def _transform(
        self,
        profiles:      List[PatientProfile],
        state_vectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Applies the already fitted encoder and scaler to new patient data.

        Calling transform (not fit_transform) here prevents test-set information
        from leaking into the feature normalisation, preserving evaluation validity.
        """

        subtypes        = np.array([[p.aphasia_subtype] for p in profiles])
        wab_aq          = np.array([p.wab_aq for p in profiles]).reshape(-1, 1)
        subtype_encoded = self._encoder.transform(subtypes)

        numeric = [wab_aq]
        if state_vectors is not None:
            numeric.append(self._extract_discourse_means(state_vectors))

        numeric_scaled = self._scaler.transform(np.hstack(numeric))
        return np.hstack([subtype_encoded, numeric_scaled]).astype(np.float32)

    def compute_inertias(self, X: np.ndarray, max_k: int) -> List[float]:
        """Runs KMeans for each k from 1 to max_k and records how tight the clusters are.
        Lower inertia means patients within a cluster are closer together.

        Inertia is the within-cluster sum of squared distances to the centroid.
        These values are consumed by find_elbow_k to identify the optimal k.
        """

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
        """Picks the best number of clusters using the elbow method.
        The elbow is the point where adding more clusters stops giving much benefit.

        The second derivative of the inertia curve identifies the sharpest bend,
        which corresponds to the point of diminishing returns. The result is then
        clamped to [min_k, max_k] to enforce the clinical minimum of six clusters.
        """

        if max_k < 3:
            return min_k

        inertias      = self.compute_inertias(X, max_k)
        logger.info(f"Inertia values: {inertias}")

        deltas        = np.diff(inertias)
        # The second derivative tells us where the curve bends the most (the elbow).
        second_deltas = np.diff(deltas)

        if len(second_deltas) == 0:
            return min_k

        # argmax finds the index of the largest absolute bend; +2 corrects for the
        # two np.diff calls that each reduce the array length by one.
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
        """Returns a boolean array where True means the patient has aphasia.

        Preferring raw_subtypes over normalised profile subtypes avoids the
        edge case where a control's subtype was silently mapped to 'Other'
        during normalisation, which would cause them to slip through as aphasia.
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
        """Averages each speech metric across all tasks for every patient.
        This gives one representative score per metric rather than one per task.

        The state vector layout is: [task_0_metric_0, task_0_metric_1, ...,
        task_1_metric_0, ...], so metrics for task t start at t * N_METRICS_PER_TASK.
        Averaging across tasks produces a compact summary that is less sensitive
        to which discourse tasks were present in a given session.
        """

        N          = state_vectors.shape[0]
        n_metrics  = N_METRICS_PER_TASK
        n_tasks    = N_TASKS
        disc_means = np.zeros((N, n_metrics), dtype=np.float32)

        for m in range(n_metrics):
            # Collect the column index for metric m in each task, then average.
            dims = [t * n_metrics + m for t in range(n_tasks)]
            disc_means[:, m] = state_vectors[:, dims].mean(axis=1)

        return disc_means