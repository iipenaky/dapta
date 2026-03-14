"""
reward.py
=========
Discourse-level reward function for DAPTA.

    R = sum_i( w_i * delta_i )   where delta_i = metric_i_after - metric_i_before

For mean_surprisal the sign is flipped (lower surprisal = better).

Weights are NOT hardcoded here. They are learned via Optuna HPO in
run_rq4.py. During a trial, Optuna samples 5 raw logits, passes them
through softmax (guaranteeing sum=1, all positive), and supplies the
result as `weights` to compute_reward / compute_batch_rewards.

Metrics used (all validated against CLAN, r >= 0.78, p < 0.001):
    mlu_morphemes   r = 0.779  grammatical complexity
    ttr             r = 0.947  lexical diversity
    ndw             r = 0.940  vocabulary breadth
    n_utterances    r = 0.782  verbal output / engagement
    mean_surprisal  —          fluency proxy (inverted: lower = better)

References
----------
Stark et al. (2021). Standardising assessment of spoken discourse in aphasia.
    AJSLP, 30(1S), 491-502.
Ng et al. (1999). Policy invariance under reward transformations. ICML.
"""

import numpy as np
from typing import Dict, Optional

# Metric ordering must match state vector ordering in state_builder.py
METRIC_ORDER = [
    "mlu_morphemes",
    "ttr",
    "ndw",
    "n_utterances",
    "mean_surprisal",   # inverted in reward
]

N_METRICS = len(METRIC_ORDER)

# Index of the surprisal dimension (inverted)
_SURPRISAL_IDX = METRIC_ORDER.index("mean_surprisal")


# =============================================================================
# WEIGHT UTILITIES
# =============================================================================

def softmax_weights(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw Optuna logits → valid reward weights (sum=1, all positive).

    Optuna samples one float per metric with no constraints.
    Softmax maps them to a probability simplex — always valid, always
    differentiable. This is the standard trick for learning weights that
    must sum to 1.

    Parameters
    ----------
    logits : np.ndarray shape (N_METRICS,)
        Raw unconstrained values from Optuna.

    Returns
    -------
    np.ndarray shape (N_METRICS,) summing to 1.0
    """
    e = np.exp(logits - logits.max())   # numerically stable
    return (e / e.sum()).astype(np.float32)


def weights_from_trial(trial) -> np.ndarray:
    """
    Sample reward weights from an Optuna trial.

    Call this inside your Optuna objective function:

        weights = weights_from_trial(trial)
        env = TherapyEnv(..., reward_weights=weights)

    Each logit is sampled in [-3, 3]. Softmax keeps the effective
    range of any single weight between ~0.03 and ~0.87 for 5 metrics.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    np.ndarray shape (N_METRICS,) summing to 1.0
    """
    logits = np.array([
        trial.suggest_float(f"w_logit_{m}", -3.0, 3.0)
        for m in METRIC_ORDER
    ], dtype=np.float32)
    return softmax_weights(logits)


def weights_to_dict(weights: np.ndarray) -> Dict[str, float]:
    """Convert weight array → readable dict for logging."""
    return {m: float(w) for m, w in zip(METRIC_ORDER, weights)}


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

def compute_reward(
    state_before:    np.ndarray,
    state_after:     np.ndarray,
    weights:         np.ndarray,
    clip:            tuple = (-1.0, 1.0),
) -> float:
    """
    Compute scalar reward from a single state transition.

    Parameters
    ----------
    state_before : np.ndarray shape (N_METRICS,)
        Normalised discourse state before therapy exercise.
    state_after  : np.ndarray shape (N_METRICS,)
        Normalised discourse state after therapy exercise.
    weights      : np.ndarray shape (N_METRICS,) summing to 1.0.
        Learned reward weights from Optuna (use weights_from_trial).
    clip         : (min, max) reward clipping bounds.

    Returns
    -------
    float in [clip[0], clip[1]]
    """
    assert len(state_before) == N_METRICS, (
        f"State length {len(state_before)} != expected {N_METRICS}. "
        f"Metrics: {METRIC_ORDER}"
    )
    assert len(weights) == N_METRICS, (
        f"Weights length {len(weights)} != {N_METRICS}."
    )

    delta = state_after - state_before   # shape (N_METRICS,)

    # Flip surprisal: improvement = reduction
    signed_delta = delta.copy()
    signed_delta[_SURPRISAL_IDX] *= -1.0

    reward = float(np.dot(weights, signed_delta))
    return float(np.clip(reward, clip[0], clip[1]))


def compute_batch_rewards(
    states_before: np.ndarray,
    states_after:  np.ndarray,
    weights:       np.ndarray,
    clip:          tuple = (-1.0, 1.0),
) -> np.ndarray:
    """
    Vectorised reward computation for a batch of transitions.

    Parameters
    ----------
    states_before : np.ndarray shape (B, N_METRICS)
    states_after  : np.ndarray shape (B, N_METRICS)
    weights       : np.ndarray shape (N_METRICS,) summing to 1.0

    Returns
    -------
    np.ndarray shape (B,)
    """
    delta = states_after - states_before        # (B, N_METRICS)

    # Build signed weight vector: surprisal weight is negated
    signed_weights = weights.copy()
    signed_weights[_SURPRISAL_IDX] *= -1.0

    rewards = (delta * signed_weights).sum(axis=1)  # (B,)
    return np.clip(rewards, clip[0], clip[1]).astype(np.float32)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_weights(weights: np.ndarray) -> None:
    """Raise if weights are not a valid probability distribution."""
    if len(weights) != N_METRICS:
        raise ValueError(f"Expected {N_METRICS} weights, got {len(weights)}.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-4):
        raise ValueError(
            f"Weights must sum to 1.0, got {weights.sum():.4f}. "
            f"Use softmax_weights() or weights_from_trial()."
        )
    if (weights < 0).any():
        raise ValueError("All weights must be non-negative.")