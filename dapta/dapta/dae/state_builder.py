from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dapta.dae.metrics import DiscourseMetrics
from dapta.utils.logger import get_logger

# Module-level logger: messages will be tagged with this file's name
logger = get_logger(__name__)

# Elicitation tasks 
TASKS = [
    "cookie_theft",
    "cinderella",
    "sandwich",
    "stroke_narrative",
    "conversation",
]
# Reverse lookup: task name → positional index (0–4)
TASK_TO_IDX = {t: i for i, t in enumerate(TASKS)}

# Discourse metric names 
METRIC_NAMES = [
    "ciu_rate",            # Correct Information Units per minute
    "mc_score",            # Main Concept score (0–1)
    "mlu_morphemes",       # Mean Length of Utterance in morphemes
    "syntactic_complexity",# Proportion of utterances with complex dependencies
    "mattr",               # Moving-Average Type-Token Ratio (vocabulary diversity)
]

#  Aphasia subtypes
APHASIA_SUBTYPES = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]
# Reverse lookup: subtype name → one-hot index
SUBTYPE_TO_IDX = {s: i for i, s in enumerate(APHASIA_SUBTYPES)}

# Median WAB Aphasia Quotient (AQ) by subtype, derived from literature.
WAB_AQ_MEDIAN: Dict[str, float] = {
    "Anomic":     78.0,  # mild: near-normal fluency, word-finding difficulty
    "Conduction": 68.0,  # moderate: fluent but poor repetition
    "Wernicke":   52.0,  # moderate-severe: fluent but poor comprehension
    "Broca":      46.0,  # moderate-severe: non-fluent, effortful speech
    "Other":      55.0,  # mid-range default for unclassified subtypes
    "Global":     18.0,  # severe: affects all language modalities
}

#  State vector layout: Total: 25 + 5 + 6 + 4 + 7 = 47 dimensions
N_METRICS_PER_TASK = len(METRIC_NAMES)  # 5
N_TASKS            = len(TASKS)         # 5
N_DISCOURSE        = N_METRICS_PER_TASK * N_TASKS  # 25
N_FLAGS            = N_TASKS            # 5  (one presence flag per task)
N_SUBTYPE          = len(APHASIA_SUBTYPES)         # 6
N_STATIC           = 4   # wab_aq, wab_aq_known, surprisal, log_session_num
N_SIGNAL           = 7   # maze_rate + 5 × utt_length_std + mean_pause_ms
STATE_DIM = N_DISCOURSE + N_FLAGS + N_SUBTYPE + N_STATIC + N_SIGNAL  # 47

# Convenience slices for indexing into the state vector by block
SLICE_DISCOURSE = slice(0,  25)
SLICE_FLAGS     = slice(25, 30)
SLICE_SUBTYPE   = slice(30, 36)
SLICE_STATIC    = slice(36, 40)
SLICE_SIGNAL    = slice(40, 47)


class PatientProfile:
    """
    Stores the static clinical metadata for one participant.

    This object is created once per participant and reused across sessions.
    It is passed to PatientStateBuilder.build() to contribute the clinical
    context portion of the RL state vector.

    Parameters
    ----------
    participant_id  : Unique string identifier for the participant.
    aphasia_subtype : Clinical aphasia classification (e.g. "Broca", "Anomic").
                      Variant spellings and aliases are normalised automatically.
    wab_aq          : Western Aphasia Battery Aphasia Quotient (0–100).
                      If None, the subtype-level median from WAB_AQ_MEDIAN is
                      used as a best-guess estimate and wab_aq_known is set False.
    session_number  : Which therapy / assessment session this is (1-indexed).
                      Used to compute the log-session feature so the model can
                      account for practice effects over time.
    """

    def __init__(
        self,
        participant_id:  str,
        aphasia_subtype: str = "Other",
        wab_aq: Optional[float] = None,
        session_number:  int = 1,
    ) -> None:
        self.participant_id  = participant_id
        # Normalise the subtype string to a canonical form (e.g. "brocas" → "Broca")
        self.aphasia_subtype = self.normalise_subtype(aphasia_subtype)
        # Clamp session number to at least 1 to avoid log(0) issues downstream
        self.session_number  = max(1, int(session_number))

        # Control participants are identified by subtype keyword or by AQ ≥ 93.8
        # (the published clinical cut-off for "not aphasic"). Controls should be
        # excluded from RL training since they have no deficit to remediate.
        raw_lower = aphasia_subtype.lower().strip()
        self.is_control = (
            "notaphasic" in raw_lower
            or "control"  in raw_lower
            or (wab_aq is not None and float(wab_aq) >= 93.8)
        )

        if wab_aq is None:
            # No measured AQ: use the subtype median as a plausible imputation
            self.wab_aq       = WAB_AQ_MEDIAN[self.aphasia_subtype]
            self.wab_aq_known = False  # flag that this value was imputed, not measured
        else:
            # Clip to [0, 100] to guard against data-entry errors
            self.wab_aq       = float(np.clip(wab_aq, 0.0, 100.0))
            self.wab_aq_known = True   # flag that this is a real clinical measurement

        if self.is_control:
            logger.info(
                f"PatientProfile: {participant_id!r} flagged as control "
                f"(subtype={aphasia_subtype!r}, wab_aq={self.wab_aq:.1f}). "
                f"Exclude from RL training."
            )

    @staticmethod
    def normalise_subtype(subtype: str) -> str:
        """
        Maps variant spellings and aliases to the canonical subtype names used
        in APHASIA_SUBTYPES. Returns "Other" for any unrecognised string.

        Examples
        --------
        "brocas"    → "Broca"
        "non-fluent" → "Broca"
        "wernickes" → "Wernicke"
        "fluent"    → "Wernicke"
        "anomia"    → "Anomic"
        """
        mapping = {
            "broca":      "Broca",    "brocas":    "Broca",
            "non-fluent": "Broca",
            "wernicke":   "Wernicke", "wernickes": "Wernicke",
            "fluent":     "Wernicke",
            "anomic":     "Anomic",   "anomia":    "Anomic",
            "conduction": "Conduction",
            "global":     "Global",
        }
        return mapping.get(subtype.lower().strip(), "Other")

    def subtype_one_hot(self) -> np.ndarray:
        """
        Returns a 6-element float32 one-hot vector encoding the aphasia subtype.
        The active index corresponds to the subtype's position in APHASIA_SUBTYPES.
        Unknown subtypes map to the "Other" index.
        """
        vec      = np.zeros(N_SUBTYPE, dtype=np.float32)
        idx      = SUBTYPE_TO_IDX.get(self.aphasia_subtype, SUBTYPE_TO_IDX["Other"])
        vec[idx] = 1.0
        return vec

    def __repr__(self) -> str:
        known = "real" if self.wab_aq_known else "imputed"
        ctrl  = ", CONTROL" if self.is_control else ""
        return (
            f"PatientProfile(id={self.participant_id!r}, "
            f"subtype={self.aphasia_subtype!r}, "
            f"wab_aq={self.wab_aq:.1f} ({known}), "
            f"session={self.session_number}{ctrl})"
        )


class SessionSignals:
    """
    Holds session-level prosodic and disfluency signals that are not captured
    by the per-task discourse metrics.

    These signals reflect the quality and fluency of the speech sample as a
    whole rather than its information content, and occupy the SIGNAL block of
    the state vector (indices 40–46).

    Parameters
    ----------
    maze_rate       : Proportion of utterances containing a disfluency repair
                      marker (repetition, reformulation, interruption).
    utt_length_std  : Dict mapping task name → standard deviation of utterance
                      lengths (in words) for that task. Measures how consistently
                      the participant produced utterances of similar length.
    mean_pause_ms   : Average inter-utterance gap in milliseconds. Longer pauses
                      may indicate word-finding difficulty or processing delay.
    """

    def __init__(
        self,
        maze_rate:      float = 0.0,
        utt_length_std: Dict[str, float] = None,
        mean_pause_ms:  float = 0.0,
    ) -> None:
        self.maze_rate      = float(maze_rate)
        self.utt_length_std = utt_length_std or {}  # default to empty if not provided
        self.mean_pause_ms  = float(mean_pause_ms)

    def std_array(self) -> np.ndarray:
        """
        Returns a float32 array of length N_TASKS where each element is the
        utterance-length standard deviation for the corresponding task.
        Tasks not present in utt_length_std are filled with 0.0.
        """
        arr = np.zeros(N_TASKS, dtype=np.float32)
        for task, val in self.utt_length_std.items():
            idx = TASK_TO_IDX.get(task)
            if idx is not None:
                arr[idx] = float(val)
        return arr


#  Scaler utilities 

def fit_nanaware_scaler(matrix: np.ndarray) -> MinMaxScaler:
    """
    Fits a MinMaxScaler on a 2-D matrix that may contain NaN values.

    Standard sklearn MinMaxScaler.fit() crashes on NaN inputs because it uses
    np.min/max internally. This function uses np.nanmin/nanmax instead, so
    columns with missing values are still scaled correctly using the range of
    the non-missing entries. The scaler's internal attributes are set manually
    to match exactly what sklearn.fit() would produce, so the returned object
    is fully compatible with transform() and save/load workflows.

    Columns where all values are NaN (data_range_ == 0) get a scale_ of 0.0,
    meaning they will always map to 0 after transformation.

    Parameters
    ----------
    matrix : 2-D float array of shape (n_samples, n_features), may contain NaN.

    Returns
    -------
    A fitted MinMaxScaler instance.
    """
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)

    scaler              = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_    = col_min
    scaler.data_max_    = col_max
    scaler.data_range_  = col_max - col_min
    # Where the range is zero (constant feature), scale to 0 to avoid divide-by-zero
    scaler.scale_       = np.where(scaler.data_range_ > 0, 1.0 / scaler.data_range_, 0.0)
    scaler.min_         = -col_min * scaler.scale_
    scaler.n_features_in_   = matrix.shape[1]
    # Count only rows that had a valid (non-NaN) value in the first column
    scaler.n_samples_seen_  = int(np.sum(~np.isnan(matrix[:, 0])))
    scaler.feature_names_in_ = None
    return scaler


def transform_nanaware(scaler: MinMaxScaler, row: np.ndarray) -> np.ndarray:
    """
    Applies a fitted MinMaxScaler to a single 1-D feature row without crashing
    on NaN values (sklearn's transform() raises on NaN by default).

    The formula is the standard MinMax transformation:
        x_scaled = x * scale_ + min_

    The result is clipped to [0, 1] to handle any values that fall slightly
    outside the training range (e.g. a test-set value larger than the training
    maximum).

    Parameters
    ----------
    scaler : A MinMaxScaler fitted by fit_nanaware_scaler().
    row    : 1-D float array of length n_features.

    Returns
    -------
    Normalised float32 array clipped to [0, 1].
    """
    out = row * scaler.scale_ + scaler.min_
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def metrics_to_array(m: DiscourseMetrics) -> np.ndarray:
    """
    Extracts the five core discourse metrics from a DiscourseMetrics object
    into a fixed-order float32 array.

    The order must match METRIC_NAMES so that the discourse block of the state
    vector is consistently laid out across all build paths.
    """
    return np.array([
        m.ciu_rate,
        m.mc_score,
        m.mlu_morphemes,
        m.syntactic_complexity,
        m.mattr,
    ], dtype=np.float32)


class PatientStateBuilder:
    """
    Converts per-session data into a fixed-length normalised state vector for
    use in downstream RL or predictive models.

    The builder must be **fitted** on the full training corpus before it can
    produce normalised states for individual sessions. Fitting computes the
    per-feature min/max ranges used to scale each block of the state vector to
    [0, 1]. Two separate scalers are maintained:

      - _discourse_scaler : applied to the 25-dimensional discourse block
      - _static_scaler    : applied to the 11-dimensional static+signal block

    Once fitted, the scalers can be saved to disk and reloaded later so that
    inference-time states are scaled with the same ranges as training states.

    Parameters
    ----------
    scaler_path : Optional path to a .npz file for saving/loading scaler state.
                  If provided, save_scaler() is called automatically at the end
                  of fit().
    """

    def __init__(self, scaler_path: Optional[Union[str, Path]] = None) -> None:
        self.scaler_path         = Path(scaler_path) if scaler_path else None
        self._discourse_scaler: Optional[MinMaxScaler] = None
        self._static_scaler:    Optional[MinMaxScaler] = None
        # Flag that gates build(): prevents producing unnormalised states
        self._fitted             = False

    def fit(
        self,
        all_task_metrics: List[Dict[str, DiscourseMetrics]],
        all_surprisals:   List[float],
        all_profiles:     List[PatientProfile],
        all_signals:      Optional[List[SessionSignals]] = None,
    ) -> "PatientStateBuilder":
        """
        Fits the internal scalers on a corpus of training sessions.

        Each element of the input lists corresponds to one session. Lists must
        all be the same length. Missing tasks within a session are represented
        as NaN in the discourse matrix so they do not corrupt the per-feature
        min/max calculation.

        Parameters
        ----------
        all_task_metrics : One dict per session mapping task name → DiscourseMetrics.
        all_surprisals   : One mean-surprisal float per session (from the LM scorer).
        all_profiles     : One PatientProfile per session.
        all_signals      : One SessionSignals per session. Defaults to all-zero
                           signals if not provided.

        Returns
        -------
        self  (for method chaining)
        """
        if not all_task_metrics:
            raise ValueError("Cannot fit on an empty training set.")

        n = len(all_task_metrics)
        # Default to zero-valued signals if not supplied
        if all_signals is None:
            all_signals = [SessionSignals() for _ in range(n)]

        # Build the (n_sessions × 25) discourse matrix.
        # Absent tasks are filled with NaN so the scaler ignores those slots
        # when computing per-feature min/max ranges.
        discourse_matrix = np.stack([
            self._build_discourse_block(tm, nan_for_absent=True)
            for tm in all_task_metrics
        ])
        # Build the (n_sessions × 11) static+signal matrix
        static_matrix = np.stack([
            self._build_static_signal_raw(p, s, sup)
            for p, s, sup in zip(all_profiles, all_signals, all_surprisals)
        ])

        # Fit one scaler per block so each block is independently normalised
        self._discourse_scaler = fit_nanaware_scaler(discourse_matrix)
        self._static_scaler    = fit_nanaware_scaler(static_matrix)
        self._fitted           = True

        logger.info(
            f"PatientStateBuilder fitted on {n} transcripts. "
            f"State dim = {STATE_DIM}."
        )
        # Persist scalers immediately if a path was configured
        if self.scaler_path:
            self.save_scaler()
        return self

    def build(
        self,
        task_metrics: Dict[str, DiscourseMetrics],
        surprisal:    float,
        profile:      PatientProfile,
        signals:      Optional[SessionSignals] = None,
    ) -> np.ndarray:
        """
        Constructs the normalised state vector for a single session.

        The vector has five concatenated blocks (total length = STATE_DIM = 47):
          [0:25]  Discourse metrics: normalised, absent tasks zeroed out
          [25:30] Task presence flags: 1.0 if task was collected, 0.0 if not
          [30:36] Aphasia subtype one-hot vector
          [36:47] Static + signal features: normalised

        Must be called after fit() or load_scaler().

        Parameters
        ----------
        task_metrics : Dict mapping task name → DiscourseMetrics for this session.
        surprisal    : Mean LM surprisal score for this session.
        profile      : PatientProfile for this participant.
        signals      : Optional session-level prosodic/disfluency signals.

        Returns
        -------
        float32 array of shape (STATE_DIM,) = (47,).
        """
        if not self._fitted:
            raise RuntimeError(
                "PatientStateBuilder has not been fitted. "
                "Call .fit() on training data first."
            )
        if signals is None:
            signals = SessionSignals()

        # Build and normalise the 25-dim discourse block
        raw_disc  = self._build_discourse_block(task_metrics, nan_for_absent=False)
        norm_disc = transform_nanaware(self._discourse_scaler, raw_disc)
        # Explicitly zero out absent task slots after scaling so scaling noise
        # from a zero fill-value does not pollute the vector
        for task, idx in TASK_TO_IDX.items():
            if task not in task_metrics:
                s = idx * N_METRICS_PER_TASK
                norm_disc[s: s + N_METRICS_PER_TASK] = 0.0

        # Binary presence flags: not scaled, already in {0.0, 1.0}
        flags      = self._build_flags(task_metrics)
        # 6-element one-hot for aphasia subtype: not scaled
        subtype_oh = profile.subtype_one_hot()

        # Build and normalise the 11-dim static+signal block
        raw_ss  = self._build_static_signal_raw(profile, signals, surprisal)
        norm_ss = transform_nanaware(self._static_scaler, raw_ss)

        # Concatenate all blocks in the canonical order
        state = np.concatenate([norm_disc, flags, subtype_oh, norm_ss])
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
        """
        Builds state vectors for multiple sessions in one call.

        Equivalent to calling build() in a loop and stacking the results.
        All input lists must be the same length.

        Returns
        -------
        float32 array of shape (n_sessions, STATE_DIM).
        """
        if all_signals is None:
            all_signals = [SessionSignals() for _ in profiles]
        return np.stack([
            self.build(tm, s, p, sig)
            for tm, s, p, sig in zip(
                all_task_metrics, surprisals, profiles, all_signals
            )
        ])

    def build_trajectory(
        self,
        session_task_metrics: List[Dict[str, DiscourseMetrics]],
        session_surprisals:   List[float],
        profile:              PatientProfile,
        session_signals:      Optional[List[SessionSignals]] = None,
    ) -> np.ndarray:
        """
        Builds a time-ordered sequence of state vectors for a single participant
        across multiple sessions (a trajectory).

        The same PatientProfile is used for every session: only the discourse
        metrics, surprisal, and session signals vary over time.

        Returns
        -------
        float32 array of shape (n_sessions, STATE_DIM).
        """
        if session_signals is None:
            session_signals = [SessionSignals() for _ in session_surprisals]
        return np.stack([
            self.build(tm, s, profile, sig)
            for tm, s, sig in zip(
                session_task_metrics, session_surprisals, session_signals
            )
        ])

    #  Private helpers 

    @staticmethod
    def _build_discourse_block(
        task_metrics:  Dict[str, DiscourseMetrics],
        nan_for_absent: bool = False,
    ) -> np.ndarray:
        """
        Assembles the 25-element raw discourse block.

        Each task occupies a contiguous 5-element slice (task_index × 5 : +5).
        Tasks present in task_metrics are filled with their metric values;
        absent tasks are filled with NaN (during fitting, so the scaler ignores
        them) or 0.0 (during inference, later zeroed after scaling).

        Unknown task names are logged and skipped rather than raising an error.
        """
        fill  = np.nan if nan_for_absent else 0.0
        block = np.full(N_DISCOURSE, fill, dtype=np.float32)
        for task, metrics in task_metrics.items():
            idx = TASK_TO_IDX.get(task)
            if idx is None:
                logger.warning(f"Unknown task '{task}': skipping.")
                continue
            s = idx * N_METRICS_PER_TASK
            block[s: s + N_METRICS_PER_TASK] = metrics_to_array(metrics)
        return block

    @staticmethod
    def _build_flags(task_metrics: Dict[str, DiscourseMetrics]) -> np.ndarray:
        """
        Builds a 5-element binary vector indicating which tasks were collected.

        A value of 1.0 means data for that task is present in this session;
        0.0 means it is absent. These flags let the model distinguish between
        "task not done" (flag = 0, metrics = 0) and "task done but metrics
        happen to be zero" (flag = 1, metrics = 0).
        """
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
        """
        Assembles the raw (unscaled) 11-element static+signal feature vector.

        Layout:
          [0]    wab_aq         : AQ score (0–100); imputed median if unknown
          [1]    wab_aq_known   : 1.0 if measured, 0.0 if imputed
          [2]    surprisal      : mean LM NLL per token for this session
          [3]    log_session_num: log1p(session_number - 1); 0.0 for session 1,
                                   grows slowly for later sessions (captures
                                   practice / familiarity effects without
                                   unbounded growth)
          [4]    maze_rate      : proportion of disfluent utterances
          [5:10] utt_length_std : per-task utterance-length std (5 values)
          [10]   mean_pause_ms  : average inter-utterance pause in ms
        """
        return np.array([
            profile.wab_aq,
            float(profile.wab_aq_known),
            float(surprisal),
            # log1p(session - 1): session 1 → 0.0, session 2 → 0.69, session 10 → 2.20
            float(np.log1p(profile.session_number - 1)),
            signals.maze_rate,
            *signals.std_array(),    # unpacks 5 per-task std values
            signals.mean_pause_ms,
        ], dtype=np.float32)

    #  Persistence 

    def save_scaler(self) -> None:
        """
        Saves the fitted scaler parameters to a .npz file at scaler_path.

        Only the arrays needed to reconstruct the scalers are saved (min, max,
        scale, min_ offset). The full sklearn object is not pickled so the file
        remains portable across sklearn versions. Use load_scaler() to restore.
        """
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.scaler_path),
            # Discourse scaler parameters
            disc_min   = self._discourse_scaler.data_min_,
            disc_max   = self._discourse_scaler.data_max_,
            disc_scale = self._discourse_scaler.scale_,
            disc_min_  = self._discourse_scaler.min_,
            # Static+signal scaler parameters
            stat_min   = self._static_scaler.data_min_,
            stat_max   = self._static_scaler.data_max_,
            stat_scale = self._static_scaler.scale_,
            stat_min_  = self._static_scaler.min_,
        )
        logger.info(f"Scalers saved to {self.scaler_path}")

    def load_scaler(self) -> "PatientStateBuilder":
        """
        Restores scaler parameters from a previously saved .npz file and marks
        the builder as fitted so build() can be called immediately.

        Raises FileNotFoundError if scaler_path is not set or does not exist.

        Returns
        -------
        self  (for method chaining)
        """
        if not self.scaler_path or not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        data = np.load(str(self.scaler_path))

        def _restore(prefix: str, n_features: int) -> MinMaxScaler:
            """
            Reconstructs a MinMaxScaler from the saved arrays for one block.
            The prefix ("disc" or "stat") selects the correct keys from the file.
            """
            s = MinMaxScaler(feature_range=(0, 1))
            s.data_min_  = data[f"{prefix}_min"]
            s.data_max_  = data[f"{prefix}_max"]
            s.data_range_ = s.data_max_ - s.data_min_
            s.scale_     = data[f"{prefix}_scale"]
            s.min_       = data[f"{prefix}_min_"]
            s.n_features_in_  = n_features
            s.n_samples_seen_ = 0           # not meaningful after loading
            s.feature_names_in_ = None
            return s

        self._discourse_scaler = _restore("disc", N_DISCOURSE)
        self._static_scaler    = _restore("stat", N_STATIC + N_SIGNAL)
        self._fitted           = True
        logger.info(f"Scalers loaded from {self.scaler_path}")
        return self

    #  Introspection helpers 

    @staticmethod
    def dim_names() -> List[str]:
        """
        Returns a list of STATE_DIM human-readable dimension names, one per
        element of the state vector, in canonical order.

        Useful for building pandas DataFrames, logging feature importances, or
        debugging unexpected values in a specific state dimension. The list is
        asserted to have exactly STATE_DIM entries so any future layout change
        that forgets to update this method will raise immediately.
        """
        names = []
        # Discourse block: "{task}__{metric}" for every (task, metric) pair
        for task in TASKS:
            for metric in METRIC_NAMES:
                names.append(f"{task}__{metric}")
        # Flags block: "flag__{task}"
        for task in TASKS:
            names.append(f"flag__{task}")
        # Subtype one-hot: "subtype__{subtype_lowercase}"
        for subtype in APHASIA_SUBTYPES:
            names.append(f"subtype__{subtype.lower()}")
        # Static features
        names.append("wab_aq")
        names.append("wab_aq_known")
        names.append("mean_surprisal")
        names.append("log_session_num")
        # Signal features
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
        """Returns the total number of dimensions in the state vector (47)."""
        return STATE_DIM