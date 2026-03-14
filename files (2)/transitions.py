"""
transitions.py
==============
Builds the (s, a, s') transition dataset for training the TransitionModel.

Two sources of transitions are combined:

1. Real transitions — from patients with 2+ recorded sessions in
   AphasiaBank. Consecutive session rows are treated as before/after
   a therapy block. These are sparse but grounded in real data.

2. Synthetic transitions — generated via RCT-derived effect size
   priors (TransitionModel.prior_predict) for single-session patients.
   Actions are sampled weighted by evidence level: Level_I exercises
   (RCT evidence) are sampled at 2x the rate of Level_II exercises.
   This is the honest way to use priors — concentrate synthetic data
   on the exercises we actually have RCT support for.

The combined dataset is saved to data/processed/transitions.csv for
inspection and reproducibility.

Thesis note
-----------
AphasiaBank is cross-sectional by design. The real transition subset
is small; the majority of training data is prior-derived. This is
acknowledged as a limitation — the transition model is largely
prior-driven, anchored by whatever real longitudinal data exists.
The framework contribution does not depend on the transition model
being perfectly calibrated to individual patients.
"""

from pathlib import Path
from typing  import Optional, Tuple

import numpy  as np
import pandas as pd

from action_space import (
    N_ACTIONS,
    ACTION_ID_TO_NAME,
    THERAPY_EXERCISES,
)
from reward import METRIC_ORDER, N_METRICS

# Minimum sessions for a patient to contribute real triples
MIN_SESSIONS = 2

# Synthetic triples generated per single-session patient
AUGMENT_PER_PATIENT = 5

# Sampling weights by evidence level
_LEVEL_I_WEIGHT  = 2.0
_LEVEL_II_WEIGHT = 1.0


# =============================================================================
# ACTION SAMPLING WEIGHTS
# =============================================================================

def _build_action_probs() -> np.ndarray:
    """
    Build action sampling probability vector weighted by evidence level.
    Level_I actions (RCT) get 2x weight over Level_II actions.
    """
    weights = np.array([
        _LEVEL_I_WEIGHT if ex.evidence_level == "Level_I" else _LEVEL_II_WEIGHT
        for ex in sorted(THERAPY_EXERCISES, key=lambda e: e.action_id)
    ], dtype=np.float32)
    return weights / weights.sum()


_ACTION_PROBS = _build_action_probs()


# =============================================================================
# THERAPY TYPE → ACTION ID MAPPING
# =============================================================================

def map_therapy_type(therapy_str: str) -> int:
    """
    Map a free-text therapy label from AphasiaBank to an action_id.
    Falls back to action 11 (free_conversation_prompting) if unrecognised,
    since AphasiaBank's picture description tasks are functionally closest
    to open conversation prompting.
    """
    if not isinstance(therapy_str, str):
        return 11
    s = therapy_str.lower().strip()
    mapping = {
        "sfa":                  0,
        "semantic feature":     0,
        "phonological":         1,
        "cueing":               1,
        "sentence svo":         2,
        "svo":                  2,
        "sentence complex":     3,
        "tuf":                  3,
        "treatment of":         3,
        "cilt":                 4,
        "constraint":           4,
        "script":               5,
        "story":                6,
        "retell":               6,
        "conversation partner": 7,
        "reading":              8,
        "writing":              9,
        "word picture":         10,
        "picture match":        10,
        "free conversation":    11,
    }
    for key, action_id in mapping.items():
        if key in s:
            return action_id
    return 11


# =============================================================================
# REAL TRANSITIONS FROM LONGITUDINAL PATIENTS
# =============================================================================

def build_real_triples(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (s, a, s') triples from patients with >= MIN_SESSIONS sessions.

    Parameters
    ----------
    df : state_vectors DataFrame. Must contain METRIC_ORDER columns,
         participant_id, and cluster_id. Optionally: session_index,
         therapy_type.

    Returns
    -------
    states      : np.ndarray shape (N, N_METRICS)
    actions     : np.ndarray shape (N,)  int64
    next_states : np.ndarray shape (N, N_METRICS)
    """
    sort_col = "session_index" if "session_index" in df.columns else None

    s_list, a_list, ns_list = [], [], []

    for pid, group in df.groupby("participant_id"):
        if len(group) < MIN_SESSIONS:
            continue
        if sort_col:
            group = group.sort_values(sort_col)
        else:
            group = group.reset_index(drop=True)

        rows = group[METRIC_ORDER].values.astype(np.float32)

        if "therapy_type" in group.columns:
            action_ids = [
                map_therapy_type(t)
                for t in group["therapy_type"].values[:-1]
            ]
        else:
            action_ids = [11] * (len(rows) - 1)

        for i in range(len(rows) - 1):
            s_list.append(rows[i])
            a_list.append(action_ids[i])
            ns_list.append(rows[i + 1])

    if not s_list:
        print("  No longitudinal patients found — "
              "all transitions will come from RCT priors.")
        empty = np.zeros((0, N_METRICS), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), empty

    states      = np.array(s_list,  dtype=np.float32)
    actions     = np.array(a_list,  dtype=np.int64)
    next_states = np.array(ns_list, dtype=np.float32)

    print(f"  Real triples from longitudinal patients: {len(states)}")
    return states, actions, next_states


# =============================================================================
# SYNTHETIC TRANSITIONS FROM RCT PRIORS
# =============================================================================

def build_prior_triples(
    df:             pd.DataFrame,
    transition_model,               # TransitionModel — used unfitted
    n_per_patient:  int  = AUGMENT_PER_PATIENT,
    seed:           int  = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic (s, a, s') triples for single-session patients
    using TransitionModel.prior_predict().

    Actions are sampled proportional to evidence level (Level_I at 2x).

    Parameters
    ----------
    df               : full state_vectors DataFrame
    transition_model : any TransitionModel instance (prior_predict does
                       not require fitting)
    n_per_patient    : synthetic triples per single-session patient
    seed             : RNG seed for reproducibility

    Returns
    -------
    states, actions, next_states — same shapes as build_real_triples
    """
    session_counts  = df.groupby("participant_id").size()
    single_pids     = session_counts[session_counts < MIN_SESSIONS].index

    # Only augment aphasia patients
    df_single = df[
        (df["participant_id"].isin(single_pids)) &
        (df["cluster_id"] >= 0)
    ].copy()

    if df_single.empty:
        print("  No single-session patients to augment.")
        empty = np.zeros((0, N_METRICS), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), empty

    print(f"  Generating {n_per_patient} synthetic triples for each of "
          f"{len(df_single)} single-session patients...")

    rng      = np.random.default_rng(seed)
    s_list, a_list, ns_list = [], [], []

    for _, row in df_single.iterrows():
        s = row[METRIC_ORDER].values.astype(np.float32)
        s = np.nan_to_num(s, nan=0.5)
        s = np.clip(s, 0.0, 1.0)

        for _ in range(n_per_patient):
            a  = int(rng.choice(N_ACTIONS, p=_ACTION_PROBS))
            ns = transition_model.prior_predict(s, a)
            s_list.append(s)
            a_list.append(a)
            ns_list.append(ns)

    states      = np.array(s_list,  dtype=np.float32)
    actions     = np.array(a_list,  dtype=np.int64)
    next_states = np.array(ns_list, dtype=np.float32)

    print(f"  Synthetic triples generated: {len(states)}")
    return states, actions, next_states


# =============================================================================
# COMBINED DATASET
# =============================================================================

def build_transition_dataset(
    df:               pd.DataFrame,
    transition_model,
    augment:          bool = True,
    n_per_patient:    int  = AUGMENT_PER_PATIENT,
    seed:             int  = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full (s, a, s') training dataset.

    Combines real longitudinal triples with synthetic prior-based triples
    for single-session patients.

    Parameters
    ----------
    df               : state_vectors DataFrame with cluster_id column
    transition_model : TransitionModel instance (used for prior_predict)
    augment          : if False, return only real triples
    n_per_patient    : synthetic triples per single-session patient
    seed             : RNG seed

    Returns
    -------
    states      : np.ndarray shape (N_total, N_METRICS)
    actions     : np.ndarray shape (N_total,)
    next_states : np.ndarray shape (N_total, N_METRICS)
    """
    print("\n  Building real transition triples...")
    r_s, r_a, r_ns = build_real_triples(df)

    if augment:
        print("\n  Building synthetic (prior) transition triples...")
        p_s, p_a, p_ns = build_prior_triples(
            df, transition_model, n_per_patient, seed
        )
    else:
        print("\n  Augmentation disabled — using real triples only.")
        empty = np.zeros((0, N_METRICS), dtype=np.float32)
        p_s, p_a, p_ns = empty, np.zeros((0,), dtype=np.int64), empty

    if len(r_s) == 0 and len(p_s) == 0:
        raise RuntimeError(
            "No transition triples available. Cannot train TransitionModel.\n"
            "Check that state_vectors.csv has multiple sessions per patient,\n"
            "or enable augmentation."
        )

    if len(r_s) == 0:
        states, actions, next_states = p_s, p_a, p_ns
    elif len(p_s) == 0:
        states, actions, next_states = r_s, r_a, r_ns
    else:
        states      = np.concatenate([r_s,  p_s],  axis=0)
        actions     = np.concatenate([r_a,  p_a],  axis=0)
        next_states = np.concatenate([r_ns, p_ns], axis=0)

    n_real = len(r_s)
    n_syn  = len(p_s)
    print(f"\n  Transition dataset: {len(states)} total triples "
          f"({n_real} real, {n_syn} synthetic)")

    return states, actions, next_states


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_transitions(
    states:      np.ndarray,
    actions:     np.ndarray,
    next_states: np.ndarray,
    out_path:    str,
) -> None:
    """Save transitions to CSV for inspection and reproducibility."""
    rows = []
    for i in range(len(states)):
        row = {f"s_{m}": float(states[i, j])
               for j, m in enumerate(METRIC_ORDER)}
        row["action"]      = int(actions[i])
        row["action_name"] = ACTION_ID_TO_NAME[int(actions[i])]
        for j, m in enumerate(METRIC_ORDER):
            row[f"ns_{m}"] = float(next_states[i, j])
        rows.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Transitions saved → {out_path}")


def load_transitions(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a previously saved transitions CSV."""
    df      = pd.read_csv(csv_path)
    s_cols  = [f"s_{m}"  for m in METRIC_ORDER]
    ns_cols = [f"ns_{m}" for m in METRIC_ORDER]

    states      = df[s_cols].values.astype(np.float32)
    actions     = df["action"].values.astype(np.int64)
    next_states = df[ns_cols].values.astype(np.float32)

    print(f"  Loaded {len(states)} transitions from {csv_path}")
    return states, actions, next_states