"""
dae/state_builder.py
---------------------
Builds the normalised patient state vector used by the RL agent.

State vector s = [ciu_rate, mc_score, mlu_morphemes, ttr, syntactic_complexity,
                  mean_surprisal, aphasia_subtype_enc (×6 one-hot), wab_aq, months_post_onset]

Dimensions: 6 discourse metrics + 6 subtype one-hot + 2 static = 14-dim total
(static features appended here; MDP uses full 14-dim state)

References
----------
Fromm et al. (2020). Seminars in Speech and Language, 41(1), 10–19.
Fridriksson & Hillis (2021). Journal of Stroke, 23(2), 183–201.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dapta.dae.metrics import DiscourseMetrics
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Discourse metric names (first 6 dims of state vector)
DISCOURSE_METRIC_NAMES = [
    "ciu_rate",
    "mc_score",
    "mlu_morphemes",
    "ttr",
    "syntactic_complexity",
    "mean_surprisal",
]

# Aphasia subtype order for one-hot encoding
APHASIA_SUBTYPES = [
    "Broca",
    "Wernicke",
    "Anomic",
    "Conduction",
    "Global",
    "Other",
]

SUBTYPE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(APHASIA_SUBTYPES)}

# Full state dimension
N_DISCOURSE = 6
N_SUBTYPE = len(APHASIA_SUBTYPES)    # 6
N_STATIC = 2                         # wab_aq, months_post_onset
STATE_DIM = N_DISCOURSE + N_SUBTYPE + N_STATIC   # = 14


# ---------------------------------------------------------------------------
# Patient metadata
# ---------------------------------------------------------------------------

class PatientProfile:
    """
    Static patient information used to augment the discourse state vector.

    Parameters
    ----------
    participant_id  : Unique participant identifier
    aphasia_subtype : Aphasia subtype string (one of APHASIA_SUBTYPES)
    wab_aq          : Western Aphasia Battery Aphasia Quotient [0, 100]
    months_post_onset: Months since stroke onset
    """

    def __init__(
        self,
        participant_id: str,
        aphasia_subtype: str = "Other",
        wab_aq: float = 50.0,
        months_post_onset: float = 12.0,
    ) -> None:
        self.participant_id = participant_id
        self.aphasia_subtype = self._normalise_subtype(aphasia_subtype)
        self.wab_aq = float(np.clip(wab_aq, 0.0, 100.0))
        self.months_post_onset = float(max(months_post_onset, 0.0))

    @staticmethod
    def _normalise_subtype(subtype: str) -> str:
        """Map common variant names to canonical subtype labels."""
        mapping = {
            "broca": "Broca", "brocas": "Broca", "non-fluent": "Broca",
            "wernicke": "Wernicke", "wernickes": "Wernicke", "fluent": "Wernicke",
            "anomic": "Anomic", "anomia": "Anomic",
            "conduction": "Conduction",
            "global": "Global",
        }
        return mapping.get(subtype.lower().strip(), "Other")

    def subtype_one_hot(self) -> np.ndarray:
        """Return a 6-dimensional one-hot vector for aphasia subtype."""
        vec = np.zeros(N_SUBTYPE, dtype=np.float32)
        idx = SUBTYPE_TO_IDX.get(self.aphasia_subtype, SUBTYPE_TO_IDX["Other"])
        vec[idx] = 1.0
        return vec


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

class PatientStateBuilder:
    """
    Assembles and normalises the patient state vector for a given session.

    Normalisation: min-max scaler fitted on the training set.
    The fitted scaler is saved/loaded to ensure consistent normalisation
    across training and evaluation.

    Parameters
    ----------
    scaler_path : Where to save/load the fitted scaler (numpy .npz)
    """

    def __init__(self, scaler_path: Optional[str | Path] = None) -> None:
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self._discourse_scaler: Optional[MinMaxScaler] = None
        self._static_scaler: Optional[MinMaxScaler] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        all_metrics: List[DiscourseMetrics],
        all_surprisals: List[float],
        all_profiles: List[PatientProfile],
    ) -> "PatientStateBuilder":
        """
        Fit min-max scalers on the training set.

        Parameters
        ----------
        all_metrics    : DiscourseMetrics for every training transcript
        all_surprisals : Mean surprisal for every training transcript
        all_profiles   : PatientProfile for every training participant
        """
        # Build raw discourse matrix (N, 6)
        discourse_matrix = np.array([
            [
                m.ciu_rate, m.mc_score, m.mlu_morphemes,
                m.ttr, m.syntactic_complexity, s,
            ]
            for m, s in zip(all_metrics, all_surprisals)
        ], dtype=np.float32)

        # Build raw static matrix (N, 2)
        static_matrix = np.array([
            [p.wab_aq, p.months_post_onset]
            for p in all_profiles
        ], dtype=np.float32)

        self._discourse_scaler = MinMaxScaler(feature_range=(0, 1))
        self._discourse_scaler.fit(discourse_matrix)

        self._static_scaler = MinMaxScaler(feature_range=(0, 1))
        self._static_scaler.fit(static_matrix)

        self._fitted = True
        logger.info("PatientStateBuilder fitted on training data.")

        if self.scaler_path:
            self._save_scaler()

        return self

    # ------------------------------------------------------------------
    # Building state vectors
    # ------------------------------------------------------------------

    def build(
        self,
        metrics: DiscourseMetrics,
        surprisal: float,
        profile: PatientProfile,
    ) -> np.ndarray:
        """
        Build a normalised 14-dimensional state vector.

        Layout
        ------
        [0:6]   Normalised discourse metrics
        [6:12]  Aphasia subtype one-hot
        [12:14] Normalised static features (WAB-AQ, months post-onset)

        Returns
        -------
        np.ndarray, shape (14,), dtype float32
        """
        if not self._fitted:
            raise RuntimeError(
                "StateBuilder not fitted. Call .fit() on training data first."
            )

        # Discourse part
        raw_discourse = np.array([[
            metrics.ciu_rate, metrics.mc_score, metrics.mlu_morphemes,
            metrics.ttr, metrics.syntactic_complexity, surprisal,
        ]], dtype=np.float32)
        norm_discourse = self._discourse_scaler.transform(raw_discourse)[0]

        # Subtype one-hot
        subtype_ohe = profile.subtype_one_hot()

        # Static part
        raw_static = np.array([[profile.wab_aq, profile.months_post_onset]], dtype=np.float32)
        norm_static = self._static_scaler.transform(raw_static)[0]

        return np.concatenate([norm_discourse, subtype_ohe, norm_static]).astype(np.float32)

    def build_batch(
        self,
        metrics_list: List[DiscourseMetrics],
        surprisals: List[float],
        profiles: List[PatientProfile],
    ) -> np.ndarray:
        """
        Build a batch of state vectors. Returns shape (N, STATE_DIM).
        """
        return np.stack([
            self.build(m, s, p)
            for m, s, p in zip(metrics_list, surprisals, profiles)
        ])

    # ------------------------------------------------------------------
    # State trajectory (for GRU history encoder in PRTA)
    # ------------------------------------------------------------------

    def build_trajectory(
        self,
        session_metrics: List[DiscourseMetrics],
        session_surprisals: List[float],
        profile: PatientProfile,
    ) -> np.ndarray:
        """
        Build a sequence of state vectors for a patient across sessions.

        Returns
        -------
        np.ndarray, shape (T, STATE_DIM)  where T = number of sessions
        """
        return np.stack([
            self.build(m, s, profile)
            for m, s in zip(session_metrics, session_surprisals)
        ])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_scaler(self) -> None:
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.scaler_path),
            disc_min=self._discourse_scaler.data_min_,
            disc_max=self._discourse_scaler.data_max_,
            stat_min=self._static_scaler.data_min_,
            stat_max=self._static_scaler.data_max_,
        )
        logger.info(f"Scaler saved to {self.scaler_path}")

    def load_scaler(self) -> "PatientStateBuilder":
        if self.scaler_path is None or not self.scaler_path.exists():
            raise FileNotFoundError(
                f"No scaler found at {self.scaler_path}. "
                "Run fit() on training data first."
            )
        data = np.load(str(self.scaler_path))
        self._discourse_scaler = MinMaxScaler()
        self._discourse_scaler.fit(np.stack([data["disc_min"], data["disc_max"]]))
        self._discourse_scaler.data_min_ = data["disc_min"]
        self._discourse_scaler.data_max_ = data["disc_max"]
        self._discourse_scaler.scale_ = 1.0 / (data["disc_max"] - data["disc_min"] + 1e-8)
        self._discourse_scaler.data_range_ = data["disc_max"] - data["disc_min"]

        self._static_scaler = MinMaxScaler()
        self._static_scaler.fit(np.stack([data["stat_min"], data["stat_max"]]))
        self._static_scaler.data_min_ = data["stat_min"]
        self._static_scaler.data_max_ = data["stat_max"]
        self._static_scaler.scale_ = 1.0 / (data["stat_max"] - data["stat_min"] + 1e-8)
        self._static_scaler.data_range_ = data["stat_max"] - data["stat_min"]

        self._fitted = True
        logger.info(f"Scaler loaded from {self.scaler_path}")
        return self
