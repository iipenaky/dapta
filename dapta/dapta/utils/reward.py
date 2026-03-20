"""
Discourse-level reward function for DAPTA.

Computes reward from the 47-dimensional patient state vector.
Averages discourse metric improvements across all tasks the patient
performed, using task presence flags (dims 25-29) to identify
valid tasks.

Reward formula:
    R = w1*mean(ΔCIU) + w2*mean(ΔMC) + w3*mean(ΔMLU)
      + w4*mean(ΔSynComp) + w5*mean(ΔMATTR) − w6*ΔSurprisal

Weights are set a priori based on clinical validity (Stark et al., 2021).
MATTR retained as length-robust lexical diversity measure despite lower
CLAN correlation (r=0.683), reflecting construct difference not error.

References
----------
Stark et al. (2021). Standardising assessment of spoken discourse in aphasia.
    AJSLP, 30(1S), 491-502.
Ng et al. (1999). Policy invariance under reward transformations. ICML.
"""

import numpy as np
from typing import Optional

from dapta.dae.state_builder import (
    N_METRICS_PER_TASK,
    N_TASKS,
    SLICE_FLAGS,
    SLICE_STATIC,
)

# Metric indices within each task block
# Layout per task: [ciu_rate, mc_score, mlu_morphemes, syncomp, mattr]
_CIU_IDX   = 0
_MC_IDX    = 1
_MLU_IDX   = 2
_SYN_IDX   = 3
_MATTR_IDX = 4

# Surprisal dim within static block (offset 2 from static block start)
_SURPRISAL_OFFSET = 2

# Reward weights — must sum to 1.0
_W_CIU    = 0.35   # Most direct measure of communicative informativeness [34, 42]
_W_MC     = 0.25   # Narrative completeness [38, 42]
_W_MLU    = 0.10   # Utterance structural complexity [42]
_W_SYN    = 0.10   # Syntactic complexity [42]
_W_MATTR  = 0.05   # Lexical diversity (length-robust TTR) [42]
_W_SURP   = 0.15   # LLM surprisal — inverse metric [13, 52]

assert abs(_W_CIU + _W_MC + _W_MLU + _W_SYN + _W_MATTR + _W_SURP - 1.0) < 1e-4

# Regression penalty applied if CIU or MC decreases
REGRESSION_PENALTY = -0.2


def compute_reward(
    state_before: np.ndarray,
    state_after:  np.ndarray,
    clip:         tuple = (-1.0, 1.0),
) -> float:
    """
    Compute discourse-level reward from a 47-dim state transition.

    Averages improvement across all tasks the patient performed,
    identified by task presence flags (SLICE_FLAGS).

    Parameters
    ----------
    state_before : np.ndarray shape (47,)
    state_after  : np.ndarray shape (47,)
    clip         : (min, max) reward clipping

    Returns
    -------
    float
    """
    ciu_deltas   = []
    mc_deltas    = []
    mlu_deltas   = []
    syn_deltas   = []
    mattr_deltas = []

    flags_start = SLICE_FLAGS.start

    for t in range(N_TASKS):
        # Check task presence flag
        if state_after[flags_start + t] < 0.5:
            continue

        base = t * N_METRICS_PER_TASK
        ciu_deltas.append(  float(state_after[base + _CIU_IDX]   - state_before[base + _CIU_IDX]))
        mc_deltas.append(   float(state_after[base + _MC_IDX]    - state_before[base + _MC_IDX]))
        mlu_deltas.append(  float(state_after[base + _MLU_IDX]   - state_before[base + _MLU_IDX]))
        syn_deltas.append(  float(state_after[base + _SYN_IDX]   - state_before[base + _SYN_IDX]))
        mattr_deltas.append(float(state_after[base + _MATTR_IDX] - state_before[base + _MATTR_IDX]))

    if not ciu_deltas:
        return 0.0

    # Average across present tasks
    d_ciu   = float(np.mean(ciu_deltas))
    d_mc    = float(np.mean(mc_deltas))
    d_mlu   = float(np.mean(mlu_deltas))
    d_syn   = float(np.mean(syn_deltas))
    d_mattr = float(np.mean(mattr_deltas))

    # Surprisal from static block
    surp_dim = SLICE_STATIC.start + _SURPRISAL_OFFSET
    d_surp   = float(state_after[surp_dim] - state_before[surp_dim])

    reward = (
        _W_CIU   * d_ciu
      + _W_MC    * d_mc
      + _W_MLU   * d_mlu
      + _W_SYN   * d_syn
      + _W_MATTR * d_mattr
      - _W_SURP  * d_surp   # surprisal inverted
    )

    # Regression penalty
    if d_ciu < 0 or d_mc < 0:
        reward += REGRESSION_PENALTY

    return float(np.clip(reward, clip[0], clip[1]))


def compute_batch_rewards(
    states_before: np.ndarray,
    states_after:  np.ndarray,
    clip:          tuple = (-1.0, 1.0),
) -> np.ndarray:
    """
    Vectorised reward for a batch of transitions.

    Parameters
    ----------
    states_before : (B, 47)
    states_after  : (B, 47)

    Returns
    -------
    np.ndarray shape (B,)
    """
    return np.array([
        compute_reward(sb, sa, clip)
        for sb, sa in zip(states_before, states_after)
    ], dtype=np.float32)