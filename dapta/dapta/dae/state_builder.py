from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dapta.dae.metrics import DiscourseMetrics
from dapta.utils.logger import get_logger
logger = get_logger(__name__)

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
    "mattr"
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

N_METRICS_PER_TASK = len(METRIC_NAMES)            
N_TASKS            = len(TASKS)                    
N_DISCOURSE        = N_METRICS_PER_TASK * N_TASKS  
N_FLAGS            = N_TASKS                       
N_SUBTYPE          = len(APHASIA_SUBTYPES)         
N_STATIC           = 4  
N_SIGNAL           = 7  
STATE_DIM          = N_DISCOURSE + N_FLAGS + N_SUBTYPE + N_STATIC + N_SIGNAL 

SLICE_DISCOURSE = slice(0,  25)
SLICE_FLAGS     = slice(25, 30)
SLICE_SUBTYPE   = slice(30, 36)
SLICE_STATIC    = slice(36, 40)
SLICE_SIGNAL    = slice(40, 47)

class PatientProfile:
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

class SessionSignals:

    def __init__(
        self,
        maze_rate:      float            = 0.0,
        utt_length_std: Dict[str, float] = None,
        mean_pause_ms:  float            = 0.0,
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

def _fit_nanaware_scaler(matrix: np.ndarray) -> MinMaxScaler:
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
    out = row * scaler.scale_ + scaler.min_
    return np.clip(out, 0.0, 1.0).astype(np.float32)



def _metrics_to_array(m: DiscourseMetrics) -> np.ndarray:
    return np.array([
        m.ciu_rate,
        m.mc_score,
        m.mlu_morphemes,
        m.syntactic_complexity,
        m.mattr
    ], dtype=np.float32)

class PatientStateBuilder:
    def __init__(self, scaler_path: Optional[Union[str, Path]] = None) -> None:
        self.scaler_path         = Path(scaler_path) if scaler_path else None
        self._discourse_scaler:  Optional[MinMaxScaler] = None
        self._static_scaler:     Optional[MinMaxScaler] = None
        self._fitted             = False

    def fit(
        self,
        all_task_metrics: List[Dict[str, DiscourseMetrics]],
        all_surprisals:   List[float],
        all_profiles:     List[PatientProfile],
        all_signals:      Optional[List[SessionSignals]] = None,
    ) -> "PatientStateBuilder":
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

    def build(
        self,
        task_metrics: Dict[str, DiscourseMetrics],
        surprisal:    float,
        profile:      PatientProfile,
        signals:      Optional[SessionSignals] = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "PatientStateBuilder has not been fitted. "
                "Call .fit() on training data first, or load a saved scaler."
            )

        if signals is None:
            signals = SessionSignals()

        raw_discourse  = self._build_discourse_block(task_metrics, nan_for_absent=False)
        norm_discourse = _transform_nanaware(self._discourse_scaler, raw_discourse)
        for task, idx in TASK_TO_IDX.items():
            if task not in task_metrics:
                start = idx * N_METRICS_PER_TASK
                norm_discourse[start : start + N_METRICS_PER_TASK] = 0.0

        flags = self._build_flags(task_metrics)

        subtype_ohe = profile.subtype_one_hot()

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

        if session_signals is None:
            session_signals = [SessionSignals() for _ in session_surprisals]
        return np.stack([
            self.build(tm, s, profile, sig)
            for tm, s, sig in zip(session_task_metrics, session_surprisals, session_signals)
        ])

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
        return np.array([
            profile.wab_aq,
            float(profile.wab_aq_known),
            float(surprisal),
            float(np.log1p(profile.session_number - 1)),
            signals.maze_rate,
            *signals.std_array(),  
            signals.mean_pause_ms,
        ], dtype=np.float32)

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