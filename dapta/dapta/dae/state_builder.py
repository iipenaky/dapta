"""
Builds the normalised patient state vector used by the RL agent.

State vector  s ∈ ℝ⁴³
─────────────────────────────────────────────────────────────────────────
Discourse metrics — 4 per task × 5 tasks = 20 dims
  [0:4]   cookie_theft      ciu_rate, mc_score, mlu_morphemes, syn_comp
  [4:8]   cinderella        ciu_rate, mc_score, mlu_morphemes, syn_comp
  [8:12]  sandwich          ciu_rate, mc_score*, mlu_morphemes, syn_comp
  [12:16] stroke_narrative  ciu_rate, mc_score, mlu_morphemes, syn_comp
  [16:20] conversation      ciu_rate, mc_score†, mlu_morphemes, syn_comp

  * sandwich mc_score is low-confidence (only 5 concepts)
  † conversation mc_score is always 0.0 (no concept list — not applicable)
  mattr removed — failed CLAN validation (r=0.755) and is a construct
  mismatch with CLAN's raw TTR; WPM captures rate information instead.

Task presence flags — 5 dims
  [20:25] 1.0 if the patient performed that task, 0.0 if absent.

Aphasia subtype one-hot — 6 dims
  [25:31] Broca / Wernicke / Anomic / Conduction / Global / Other

Static features — 4 dims
  [31]    wab_aq            WAB Aphasia Quotient, normalised [0, 1]
                            (imputed from subtype median if unknown)
  [32]    wab_aq_known      1.0 if real value, 0.0 if imputed
  [33]    mean_surprisal    RoBERTa surprisal (global, not per-task)
  [34]    log_session_num   log(session_number), trajectory position

Signal dims — 8 dims
  [35]    maze_rate         Repair/revision proportion (global)
  [36:41] utt_length_std    Per-task std of utterance word counts (5 dims)
  [41]    mean_pause_ms     Mean inter-utterance gap from CHAT timestamps (ms)
  [42]    wpm               Words per minute (global, counted from raw CHAT tier)
─────────────────────────────────────────────────────────────────────────

Scaler design
-------------
Discourse block (dims 0–19): NaN-aware MinMaxScaler. Absent task slots are
masked to NaN before fitting so absence-zeros do not contaminate the
per-column min/max. After transform, absent slots are re-zeroed.

Static/signal block (dims 31–42): separate MinMaxScaler.

Flags (20–24) and subtype one-hot (25–30) are in [0,1] and are never scaled.

WAB-AQ imputation medians (AphasiaBank norms)
---------------------------------------------
  Anomic     ≈ 78   Conduction ≈ 68   Wernicke ≈ 52
  Broca      ≈ 46   Global     ≈ 18   Other    ≈ 55
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dapta.dae.metrics import DiscourseMetrics
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = [
    "cookie_theft",
    "cinderella",
    "sandwich",
    "stroke_narrative",
    "conversation",
]
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}

METRIC_NAMES = [
    "ciu_rate",
    "mc_score",
    "mlu_morphemes",
    "syntactic_complexity",
]

APHASIA_SUBTYPES = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]
SUBTYPE_TO_IDX   = {s: i for i, s in enumerate(APHASIA_SUBTYPES)}

WAB_AQ_MEDIAN: Dict[str, float] = {
    "Anomic":     78.0,
    "Conduction": 68.0,
    "Wernicke":   52.0,
    "Broca":      46.0,
    "Other":      55.0,
    "Global":     18.0,
}

# Dimension bookkeeping
N_METRICS_PER_TASK = len(METRIC_NAMES)            # 4
N_TASKS            = len(TASKS)                    # 5
N_DISCOURSE        = N_METRICS_PER_TASK * N_TASKS  # 20
N_FLAGS            = N_TASKS                       # 5
N_SUBTYPE          = len(APHASIA_SUBTYPES)         # 6
N_STATIC           = 4   # wab_aq, wab_aq_known, mean_surprisal, log_session_num
N_SIGNAL           = 8   # maze_rate + 5×utt_length_std + mean_pause_ms + wpm
STATE_DIM          = N_DISCOURSE + N_FLAGS + N_SUBTYPE + N_STATIC + N_SIGNAL  # 43

# Named slices
SLICE_DISCOURSE = slice(0,  20)
SLICE_FLAGS     = slice(20, 25)
SLICE_SUBTYPE   = slice(25, 31)
SLICE_STATIC    = slice(31, 35)
SLICE_SIGNAL    = slice(35, 43)


# ---------------------------------------------------------------------------
# PatientProfile
# ---------------------------------------------------------------------------

class PatientProfile:
    """
    Static patient information used to augment the discourse state vector.

    Parameters
    ----------
    participant_id  : Unique participant identifier.
    aphasia_subtype : Aphasia subtype string (see _normalise_subtype).
    wab_aq          : WAB Aphasia Quotient [0, 100]. None triggers imputation.
    session_number  : 1-indexed session count for longitudinal ordering.
    """

    def __init__(
        self,
        participant_id:  str,
        aphasia_subtype: str             = "Other",
        wab_aq:          Optional[float] = None,
        session_number:  int             = 1,
    ) -> None:
        self.participant_id  = participant_id
        self.aphasia_subtype = self._normalise_subtype(aphasia_subtype)
        self.session_number  = max(1, int(session_number))

        if wab_aq is None:
            self.wab_aq       = WAB_AQ_MEDIAN[self.aphasia_subtype]
            self.wab_aq_known = False
        else:
            self.wab_aq       = float(np.clip(wab_aq, 0.0, 100.0))
            self.wab_aq_known = True

    @staticmethod
    def _normalise_subtype(subtype: str) -> str:
        mapping = {
            "broca":      "Broca",    "brocas":    "Broca",   "non-fluent": "Broca",
            "wernicke":   "Wernicke", "wernickes": "Wernicke","fluent":     "Wernicke",
            "anomic":     "Anomic",   "anomia":    "Anomic",
            "conduction": "Conduction",
            "global":     "Global",
        }
        return mapping.get(subtype.lower().strip(), "Other")

    def subtype_one_hot(self) -> np.ndarray:
        vec      = np.zeros(N_SUBTYPE, dtype=np.float32)
        idx      = SUBTYPE_TO_IDX.get(self.aphasia_subtype, SUBTYPE_TO_IDX["Other"])
        vec[idx] = 1.0
        return vec

    def __repr__(self) -> str:
        known = "real" if self.wab_aq_known else "imputed"
        return (
            f"PatientProfile(id={self.participant_id!r}, "
            f"subtype={self.aphasia_subtype!r}, "
            f"wab_aq={self.wab_aq:.1f} ({known}), "
            f"session={self.session_number})"
        )


# ---------------------------------------------------------------------------
# SessionSignals
# ---------------------------------------------------------------------------

class SessionSignals:
    """
    Extra per-session signals that are not part of DiscourseMetrics.

    Parameters
    ----------
    maze_rate      : Repairs as a proportion of total utterances.
                     Compute from raw CHAT before clean_utterance() strips
                     [/], [//], +/. markers.
    utt_length_std : Dict mapping task name -> std of per-utterance word
                     counts for that task. Missing tasks default to 0.0.
    mean_pause_ms  : Mean inter-utterance pause in ms from CHAT timestamps.
                     Pass 0.0 if timestamps are unavailable.
    wpm            : Words per minute (global across all tasks, computed
                     from raw CHAT tier to match CLAN's Words_Min).
    """

    def __init__(
        self,
        maze_rate:      float            = 0.0,
        utt_length_std: Dict[str, float] = None,
        mean_pause_ms:  float            = 0.0,
        wpm:            float            = 0.0,
    ) -> None:
        self.maze_rate      = float(maze_rate)
        self.utt_length_std = utt_length_std or {}
        self.mean_pause_ms  = float(mean_pause_ms)
        self.wpm            = float(wpm)

    def std_array(self) -> np.ndarray:
        """Return per-task utt_length_std as a 5-dim array (task order = TASKS)."""
        arr = np.zeros(N_TASKS, dtype=np.float32)
        for task, val in self.utt_length_std.items():
            idx = TASK_TO_IDX.get(task)
            if idx is not None:
                arr[idx] = float(val)
        return arr


# ---------------------------------------------------------------------------
# NaN-aware scaler helper
# ---------------------------------------------------------------------------

def _fit_nanaware_scaler(matrix: np.ndarray) -> MinMaxScaler:
    """
    Fit a MinMaxScaler ignoring NaN values per column.

    For each column, min/max are computed only over non-NaN rows.
    This prevents absence-zeros from contaminating the per-column range.
    """
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)

    scaler             = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_   = col_min
    scaler.data_max_   = col_max
    scaler.data_range_ = col_max - col_min
    scaler.scale_      = np.where(scaler.data_range_ > 0,
                                  1.0 / scaler.data_range_, 0.0)
    scaler.min_        = -col_min * scaler.scale_
    scaler.n_features_in_   = matrix.shape[1]
    scaler.n_samples_seen_  = int(np.sum(~np.isnan(matrix[:, 0])))
    scaler.feature_names_in_ = None
    return scaler


def _transform_nanaware(scaler: MinMaxScaler, row: np.ndarray) -> np.ndarray:
    """Apply a NaN-aware scaler to a single row (1-D array)."""
    out = row * scaler.scale_ + scaler.min_
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics_to_array(m: DiscourseMetrics) -> np.ndarray:
    """Return [ciu_rate, mc_score, mlu_morphemes, syntactic_complexity]."""
    return np.array([
        m.ciu_rate,
        m.mc_score,
        m.mlu_morphemes,
        m.syntactic_complexity,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# PatientStateBuilder
# ---------------------------------------------------------------------------

class PatientStateBuilder:
    """
    Assembles and normalises the 43-dimensional patient state vector.

    build() receives:
      - task_metrics  : dict of task name -> DiscourseMetrics
      - surprisal     : global mean RoBERTa surprisal
      - profile       : PatientProfile (subtype, WAB-AQ, session number)
      - signals       : SessionSignals (maze_rate, utt_length_std, pause, wpm)

    Normalisation
    -------------
    Discourse block (0–19): NaN-aware MinMaxScaler.
    Static + signal block (31–42): separate MinMaxScaler.
    Flags (20–24) and subtype one-hot (25–30): not scaled.
    """

    def __init__(self, scaler_path: Optional[Union[str, Path]] = None) -> None:
        self.scaler_path         = Path(scaler_path) if scaler_path else None
        self._discourse_scaler:  Optional[MinMaxScaler] = None
        self._static_scaler:     Optional[MinMaxScaler] = None
        self._fitted             = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        all_task_metrics: List[Dict[str, DiscourseMetrics]],
        all_surprisals:   List[float],
        all_profiles:     List[PatientProfile],
        all_signals:      Optional[List[SessionSignals]] = None,
    ) -> "PatientStateBuilder":
        """
        Fit scalers on the training set.

        Parameters
        ----------
        all_task_metrics : List of dicts, one per transcript.
        all_surprisals   : Global mean surprisal per transcript.
        all_profiles     : PatientProfile per transcript.
        all_signals      : SessionSignals per transcript. Defaults to zeros.
        """
        if not all_task_metrics:
            raise ValueError("Cannot fit on an empty training set.")

        n = len(all_task_metrics)
        if all_signals is None:
            all_signals = [SessionSignals() for _ in range(n)]

        discourse_matrix = np.stack([
            self._build_discourse_block(tm, nan_for_absent=True)
            for tm in all_task_metrics
        ])

        static_signal_matrix = np.stack([
            self._build_static_signal_raw(p, s, sup)
            for p, s, sup in zip(all_profiles, all_signals, all_surprisals)
        ])

        self._discourse_scaler = _fit_nanaware_scaler(discourse_matrix)
        self._static_scaler    = _fit_nanaware_scaler(static_signal_matrix)

        self._fitted = True
        logger.info(
            f"PatientStateBuilder fitted on {n} transcripts. "
            f"State dim = {STATE_DIM}."
        )
        if self.scaler_path:
            self.save_scaler()
        return self

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        task_metrics: Dict[str, DiscourseMetrics],
        surprisal:    float,
        profile:      PatientProfile,
        signals:      Optional[SessionSignals] = None,
    ) -> np.ndarray:
        """
        Build a normalised 43-dimensional state vector.

        Returns
        -------
        np.ndarray, shape (43,), dtype float32
        """
        if not self._fitted:
            raise RuntimeError(
                "PatientStateBuilder has not been fitted. "
                "Call .fit() on training data first, or load a saved scaler."
            )

        if signals is None:
            signals = SessionSignals()

        # 1. Discourse block (20,)
        raw_discourse  = self._build_discourse_block(task_metrics, nan_for_absent=False)
        norm_discourse = _transform_nanaware(self._discourse_scaler, raw_discourse)

        # 2. Re-zero absent task slots after scaling
        for task, idx in TASK_TO_IDX.items():
            if task not in task_metrics:
                start = idx * N_METRICS_PER_TASK
                norm_discourse[start : start + N_METRICS_PER_TASK] = 0.0

        # 3. Presence flags (5,)
        flags = self._build_flags(task_metrics)

        # 4. Subtype one-hot (6,)
        subtype_ohe = profile.subtype_one_hot()

        # 5. Static + signal block (12,)
        raw_ss  = self._build_static_signal_raw(profile, signals, surprisal)
        norm_ss = _transform_nanaware(self._static_scaler, raw_ss)

        state = np.concatenate([norm_discourse, flags, subtype_ohe, norm_ss])
        assert state.shape == (STATE_DIM,), (
            f"State shape mismatch: expected ({STATE_DIM},), got {state.shape}"
        )
        return state.astype(np.float32)

    def build_batch(
        self,
        all_task_metrics: List[Dict[str, DiscourseMetrics]],
        surprisals:       List[float],
        profiles:         List[PatientProfile],
        all_signals:      Optional[List[SessionSignals]] = None,
    ) -> np.ndarray:
        """Build a batch of state vectors. Returns shape (N, 43)."""
        if all_signals is None:
            all_signals = [SessionSignals() for _ in profiles]
        return np.stack([
            self.build(tm, s, p, sig)
            for tm, s, p, sig in zip(all_task_metrics, surprisals, profiles, all_signals)
        ])

    def build_trajectory(
        self,
        session_task_metrics: List[Dict[str, DiscourseMetrics]],
        session_surprisals:   List[float],
        profile:              PatientProfile,
        session_signals:      Optional[List[SessionSignals]] = None,
    ) -> np.ndarray:
        """
        Build a sequence of state vectors across sessions (for GRU encoder).
        Returns shape (T, 43) where T = number of sessions.
        """
        if session_signals is None:
            session_signals = [SessionSignals() for _ in session_surprisals]
        return np.stack([
            self.build(tm, s, profile, sig)
            for tm, s, sig in zip(session_task_metrics, session_surprisals, session_signals)
        ])

    # ------------------------------------------------------------------
    # Internal helpers
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
            start = idx * N_METRICS_PER_TASK
            block[start : start + N_METRICS_PER_TASK] = _metrics_to_array(metrics)

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
        profile:   "PatientProfile",
        signals:   "SessionSignals",
        surprisal: float,
    ) -> np.ndarray:
        """
        Raw (unscaled) 12-dim static + signal vector.

        Layout:
          [0]    wab_aq (0–100)
          [1]    wab_aq_known (0 or 1)
          [2]    mean_surprisal
          [3]    log_session_num
          [4]    maze_rate
          [5:10] utt_length_std per task (5 values)
          [10]   mean_pause_ms
          [11]   wpm
        """
        return np.array([
            profile.wab_aq,
            float(profile.wab_aq_known),
            float(surprisal),
            float(np.log1p(profile.session_number - 1)),
            signals.maze_rate,
            *signals.std_array(),   # 5 values
            signals.mean_pause_ms,
            signals.wpm,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scaler(self) -> None:
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.scaler_path),
            disc_min   = self._discourse_scaler.data_min_,
            disc_max   = self._discourse_scaler.data_max_,
            disc_scale = self._discourse_scaler.scale_,
            disc_min_  = self._discourse_scaler.min_,
            stat_min   = self._static_scaler.data_min_,
            stat_max   = self._static_scaler.data_max_,
            stat_scale = self._static_scaler.scale_,
            stat_min_  = self._static_scaler.min_,
        )
        logger.info(f"Scalers saved to {self.scaler_path}")

    def load_scaler(self) -> "PatientStateBuilder":
        if not self.scaler_path or not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        data = np.load(str(self.scaler_path))

        def _restore(prefix: str, n_features: int) -> MinMaxScaler:
            s                  = MinMaxScaler(feature_range=(0, 1))
            s.data_min_        = data[f"{prefix}_min"]
            s.data_max_        = data[f"{prefix}_max"]
            s.data_range_      = s.data_max_ - s.data_min_
            s.scale_           = data[f"{prefix}_scale"]
            s.min_             = data[f"{prefix}_min_"]
            s.n_features_in_   = n_features
            s.n_samples_seen_  = 0
            s.feature_names_in_ = None
            return s

        self._discourse_scaler = _restore("disc", N_DISCOURSE)
        self._static_scaler    = _restore("stat", N_STATIC + N_SIGNAL)
        self._fitted           = True
        logger.info(f"Scalers loaded from {self.scaler_path}")
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @staticmethod
    def dim_names() -> List[str]:
        """Return the name of every dimension in the state vector."""
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
        names.append("wpm")
        assert len(names) == STATE_DIM, (
            f"dim_names length {len(names)} != STATE_DIM {STATE_DIM}"
        )
        return names

    @staticmethod
    def state_dim() -> int:
        return STATE_DIM