"""
Builds the normalised patient state vector used by the RL agent.

State vector s = [ciu_rate, mc_score, mlu_morphemes, ttr, syntactic_complexity,
                  mean_surprisal, aphasia_subtype_enc (×6 one-hot), wab_aq, months_post_onset]

Dimensions: 6 discourse metrics + 6 subtype one-hot + 2 static = 14-dim total
(static features appended here; MDP uses full 14-dim state)
"""

from __future__ import annotations
from pathlib import Path
<<<<<<< Updated upstream
from typing import Dict, List, Optional, Tuple
=======
from typing import Dict, List, Optional, Union
>>>>>>> Stashed changes

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dapta.dae.metrics import DiscourseMetrics
from dapta.utils.logger import get_logger

logger = get_logger(__name__)



# Constants


# Discourse metric names (first 6 dims of state vector)
DISCOURSE_METRIC_NAMES = [
    "ciu_rate",
    "mc_score",
    "mlu_morphemes",
    "ttr",
    "syntactic_complexity",
<<<<<<< Updated upstream
    "mean_surprisal",
=======
    "mattr",
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
# Full state dimension
N_DISCOURSE = 6
N_SUBTYPE = len(APHASIA_SUBTYPES)    # 6
N_STATIC = 2                         # wab_aq, months_post_onset
STATE_DIM = N_DISCOURSE + N_SUBTYPE + N_STATIC   # = 14



# Patient metadata
=======
N_METRICS_PER_TASK = len(METRIC_NAMES)             # 5
N_TASKS            = len(TASKS)                    # 5
N_DISCOURSE        = N_METRICS_PER_TASK * N_TASKS  # 25
N_FLAGS            = N_TASKS                       # 5
N_SUBTYPE          = len(APHASIA_SUBTYPES)         # 6
N_STATIC           = 4
N_SIGNAL           = 7
STATE_DIM          = N_DISCOURSE + N_FLAGS + N_SUBTYPE + N_STATIC + N_SIGNAL  # 47
>>>>>>> Stashed changes



# ==================================================================
class PatientProfile:
    """
<<<<<<< Updated upstream
    Static patient information used to augment the discourse state vector.

    Parameters
    ----------
    participant_id  : Unique participant identifier
    aphasia_subtype : Aphasia subtype string (one of APHASIA_SUBTYPES)
    wab_aq          : Western Aphasia Battery Aphasia Quotient [0, 100]
    months_post_onset: Months since stroke onset
=======
    Holds per-patient clinical metadata.

    is_control is True when the participant is flagged as
    NotAphasicByWAB, 'control', or has WAB-AQ ≥ 93.8.
    Controls should be excluded from RL training/clustering
    but may still be used for DAE validation (RQ1).
>>>>>>> Stashed changes
    """

    def __init__(
        self,
<<<<<<< Updated upstream
        participant_id: str,
        aphasia_subtype: str = "Other",
        wab_aq: float = 50.0,
        months_post_onset: float = 12.0,
    ) -> None:
        self.participant_id = participant_id
        self.aphasia_subtype = self._normalise_subtype(aphasia_subtype)
        self.wab_aq = float(np.clip(wab_aq, 0.0, 100.0))
        self.months_post_onset = float(max(months_post_onset, 0.0))
=======
        participant_id:  str,
        aphasia_subtype: str           = "Other",
        wab_aq:          Optional[float] = None,
        session_number:  int           = 1,
    ) -> None:
        self.participant_id  = participant_id
        self.aphasia_subtype = self.normalise_subtype(aphasia_subtype)
        self.session_number  = max(1, int(session_number))

        # Detect control participants
        raw_lower = aphasia_subtype.lower().strip()
        self.is_control = (
            "notaphasic"  in raw_lower
            or "control"  in raw_lower
            or (wab_aq is not None and float(wab_aq) >= 93.8)
        )

        if wab_aq is None:
            self.wab_aq       = WAB_AQ_MEDIAN[self.aphasia_subtype]
            self.wab_aq_known = False
        else:
            self.wab_aq       = float(np.clip(wab_aq, 0.0, 100.0))
            self.wab_aq_known = True
>>>>>>> Stashed changes

        if self.is_control:
            logger.info(
                f"PatientProfile: {participant_id!r} flagged as control "
                f"(subtype={aphasia_subtype!r}, wab_aq={self.wab_aq:.1f}). "
                f"Exclude from RL training."
            )

    @staticmethod
<<<<<<< Updated upstream
    def _normalise_subtype(subtype: str) -> str:
        """Map common variant names to canonical subtype labels."""
        mapping = {
            "broca": "Broca", "brocas": "Broca", "non-fluent": "Broca",
            "wernicke": "Wernicke", "wernickes": "Wernicke", "fluent": "Wernicke",
            "anomic": "Anomic", "anomia": "Anomic",
=======
    def normalise_subtype(subtype: str) -> str:
        mapping = {
            "broca":      "Broca",    "brocas":    "Broca",
            "non-fluent": "Broca",
            "wernicke":   "Wernicke", "wernickes": "Wernicke",
            "fluent":     "Wernicke",
            "anomic":     "Anomic",   "anomia":    "Anomic",
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream


# State builder


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

   
    # Fitting
   
=======
    def __repr__(self) -> str:
        known = "real" if self.wab_aq_known else "imputed"
        ctrl  = ", CONTROL" if self.is_control else ""
        return (
            f"PatientProfile(id={self.participant_id!r}, "
            f"subtype={self.aphasia_subtype!r}, "
            f"wab_aq={self.wab_aq:.1f} ({known}), "
            f"session={self.session_number}{ctrl})"
        )


# ==================================================================
class SessionSignals:

    def __init__(
        self,
        maze_rate:      float              = 0.0,
        utt_length_std: Dict[str, float]   = None,
        mean_pause_ms:  float              = 0.0,
    ) -> None:
        self.maze_rate      = float(maze_rate)
        self.utt_length_std = utt_length_std or {}
        self.mean_pause_ms  = float(mean_pause_ms)

    def std_array(self) -> np.ndarray:
        arr = np.zeros(N_TASKS, dtype=np.float32)
        for task, val in self.utt_length_std.items():
            idx = TASK_TO_IDX.get(task)
            if idx is not None:
                arr[idx] = float(val)
        return arr


# ==================================================================
def fit_nanaware_scaler(matrix: np.ndarray) -> MinMaxScaler:
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)

    scaler             = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_   = col_min
    scaler.data_max_   = col_max
    scaler.data_range_ = col_max - col_min
    scaler.scale_      = np.where(scaler.data_range_ > 0,
                                  1.0 / scaler.data_range_, 0.0)
    scaler.min_        = -col_min * scaler.scale_
    scaler.n_features_in_    = matrix.shape[1]
    scaler.n_samples_seen_   = int(np.sum(~np.isnan(matrix[:, 0])))
    scaler.feature_names_in_ = None
    return scaler


def transform_nanaware(scaler: MinMaxScaler, row: np.ndarray) -> np.ndarray:
    out = row * scaler.scale_ + scaler.min_
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def metrics_to_array(m: DiscourseMetrics) -> np.ndarray:
    return np.array([
        m.ciu_rate,
        m.mc_score,
        m.mlu_morphemes,
        m.syntactic_complexity,
        m.mattr,
    ], dtype=np.float32)

>>>>>>> Stashed changes

# ==================================================================
class PatientStateBuilder:

    def __init__(self, scaler_path: Optional[Union[str, Path]] = None) -> None:
        self.scaler_path        = Path(scaler_path) if scaler_path else None
        self._discourse_scaler: Optional[MinMaxScaler] = None
        self._static_scaler:    Optional[MinMaxScaler] = None
        self._fitted            = False

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

<<<<<<< Updated upstream
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

=======
        discourse_matrix = np.stack([
            self._build_discourse_block(tm, nan_for_absent=True)
            for tm in all_task_metrics
        ])
        static_matrix = np.stack([
            self._build_static_signal_raw(p, s, sup)
            for p, s, sup in zip(all_profiles, all_signals, all_surprisals)
        ])

        self._discourse_scaler = fit_nanaware_scaler(discourse_matrix)
        self._static_scaler    = fit_nanaware_scaler(static_matrix)
        self._fitted           = True

        logger.info(
            f"PatientStateBuilder fitted on {n} transcripts. "
            f"State dim = {STATE_DIM}."
        )
>>>>>>> Stashed changes
        if self.scaler_path:
            self._save_scaler()

        return self

<<<<<<< Updated upstream
   
    # Building state vectors
   

=======
    # ------------------------------------------------------------------
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
=======
                "PatientStateBuilder has not been fitted. "
                "Call .fit() on training data first."
            )
        if signals is None:
            signals = SessionSignals()

        raw_disc  = self._build_discourse_block(task_metrics, nan_for_absent=False)
        norm_disc = transform_nanaware(self._discourse_scaler, raw_disc)
        # Zero out absent task slots after scaling
        for task, idx in TASK_TO_IDX.items():
            if task not in task_metrics:
                s = idx * N_METRICS_PER_TASK
                norm_disc[s: s + N_METRICS_PER_TASK] = 0.0

        flags      = self._build_flags(task_metrics)
        subtype_oh = profile.subtype_one_hot()

        raw_ss  = self._build_static_signal_raw(profile, signals, surprisal)
        norm_ss = transform_nanaware(self._static_scaler, raw_ss)

        state = np.concatenate([norm_disc, flags, subtype_oh, norm_ss])
        assert state.shape == (STATE_DIM,), (
            f"State shape mismatch: expected ({STATE_DIM},), got {state.shape}"
        )
        return state.astype(np.float32)
>>>>>>> Stashed changes

    # ------------------------------------------------------------------
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
<<<<<<< Updated upstream
            self.build(m, s, p)
            for m, s, p in zip(metrics_list, surprisals, profiles)
=======
            self.build(tm, s, p, sig)
            for tm, s, p, sig in zip(
                all_task_metrics, surprisals, profiles, all_signals
            )
>>>>>>> Stashed changes
        ])

   
    # State trajectory (for GRU history encoder in PRTA)
   

    def build_trajectory(
        self,
        session_metrics: List[DiscourseMetrics],
        session_surprisals: List[float],
        profile: PatientProfile,
    ) -> np.ndarray:
<<<<<<< Updated upstream
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

   
    # Persistence
   

    def _save_scaler(self) -> None:
=======
        if session_signals is None:
            session_signals = [SessionSignals() for _ in session_surprisals]
        return np.stack([
            self.build(tm, s, profile, sig)
            for tm, s, sig in zip(
                session_task_metrics, session_surprisals, session_signals
            )
        ])

    # ------------------------------------------------------------------
    @staticmethod
    def _build_discourse_block(
        task_metrics:   Dict[str, DiscourseMetrics],
        nan_for_absent: bool = False,
    ) -> np.ndarray:
        fill  = np.nan if nan_for_absent else 0.0
        block = np.full(N_DISCOURSE, fill, dtype=np.float32)
        for task, metrics in task_metrics.items():
            idx = TASK_TO_IDX.get(task)
            if idx is None:
                logger.warning(f"Unknown task '{task}' — skipping.")
                continue
            s = idx * N_METRICS_PER_TASK
            block[s: s + N_METRICS_PER_TASK] = metrics_to_array(metrics)
        return block

    @staticmethod
    def _build_flags(task_metrics: Dict[str, DiscourseMetrics]) -> np.ndarray:
        flags = np.zeros(N_FLAGS, dtype=np.float32)
        for task in task_metrics:
            idx = TASK_TO_IDX.get(task)
            if idx is not None:
                flags[idx] = 1.0
        return flags

    @staticmethod
    def _build_static_signal_raw(
        profile:   PatientProfile,
        signals:   SessionSignals,
        surprisal: float,
    ) -> np.ndarray:
        return np.array([
            profile.wab_aq,
            float(profile.wab_aq_known),
            float(surprisal),
            float(np.log1p(profile.session_number - 1)),
            signals.maze_rate,
            *signals.std_array(),
            signals.mean_pause_ms,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def save_scaler(self) -> None:
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======

    # ------------------------------------------------------------------
    @staticmethod
    def dim_names() -> List[str]:
        names = []
        for task in TASKS:
            for metric in METRIC_NAMES:
                names.append(f"{task}__{metric}")
        for task in TASKS:
            names.append(f"flag__{task}")
        for subtype in APHASIA_SUBTYPES:
            names.append(f"subtype__{subtype.lower()}")
        names.append("wab_aq")
        names.append("wab_aq_known")
        names.append("mean_surprisal")
        names.append("log_session_num")
        names.append("maze_rate")
        for task in TASKS:
            names.append(f"utt_length_std__{task}")
        names.append("mean_pause_ms")
        assert len(names) == STATE_DIM, (
            f"dim_names length {len(names)} != STATE_DIM {STATE_DIM}"
        )
        return names

    @staticmethod
    def state_dim() -> int:
        return STATE_DIM
>>>>>>> Stashed changes
