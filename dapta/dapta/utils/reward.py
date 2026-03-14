"""
Discourse-level reward function for DAPTA.

The reward is a weighted sum of improvements across six discourse metrics:
    R = w1*ΔCIU + w2*ΔMC + w3*ΔMLU + w4*ΔTTR + w5*ΔSynComp − w6*ΔSurprisal

Weights are set a priori based on the clinical validity of each metric
as an index of functional communication (Stark et al., 2021).

References
----------
Stark et al. (2021). Standardising assessment of spoken discourse in aphasia.
    AJSLP, 30(1S), 491–502.
Ng et al. (1999). Policy invariance under reward transformations.
    ICML Proceedings.
"""


import numpy as np
from typing import Dict, Optional

# Default metric weights (must sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ciu_rate": 0.35,         # Most direct measure of communicative info
    "mc_score": 0.25,         # Narrative completeness
    "mlu_morphemes": 0.10,    # Utterance length
    "ttr": 0.10,              # Lexical diversity
    "syntactic_complexity": 0.05,
    "mean_surprisal": 0.15,   # Inverse: improvement = REDUCTION in surprisal
}

# Metric ordering must match state vector ordering in state_builder.py
METRIC_ORDER = [
    "ciu_rate",
    "mc_score",
    "mlu_morphemes",
    "ttr",
    "syntactic_complexity",
    "mean_surprisal",
]

# Negative improvement penalty (applied if any primary metric worsens)
REGRESSION_PENALTY = -0.2


def compute_reward(
    state_before: np.ndarray,
    state_after: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    clip: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """
    Compute the discourse-level reward from state transition.

    Parameters
    ----------
    state_before : np.ndarray, shape (6,)
        Normalised discourse state vector before therapy exercise.
    state_after  : np.ndarray, shape (6,)
        Normalised discourse state vector after therapy exercise.
    weights      : Optional dict of metric -> weight. Uses DEFAULT_WEIGHTS if None.
    clip         : (min, max) to clip final reward.

    Returns
    -------
    float : Scalar reward in [clip[0], clip[1]]
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    _validate_weights(weights)
    assert len(state_before) == len(METRIC_ORDER), (
        f"State vector length {len(state_before)} != expected {len(METRIC_ORDER)}"
    )

    delta = state_after - state_before  # shape (6,)
    reward = 0.0

    for i, metric in enumerate(METRIC_ORDER):
        w = weights[metric]
        d = float(delta[i])

        if metric == "mean_surprisal":
            # Lower surprisal = better = positive reward
            reward += w * (-d)
        else:
            reward += w * d

    # Regression penalty: punish if CIU rate or MC score decreases
    ciu_idx = METRIC_ORDER.index("ciu_rate")
    mc_idx = METRIC_ORDER.index("mc_score")
    if delta[ciu_idx] < 0 or delta[mc_idx] < 0:
        reward += REGRESSION_PENALTY

    return float(np.clip(reward, clip[0], clip[1]))


def compute_batch_rewards(
    states_before: np.ndarray,
    states_after: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    clip: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Vectorised reward computation for a batch of transitions.

    Parameters
    ----------
    states_before : np.ndarray, shape (B, 6)
    states_after  : np.ndarray, shape (B, 6)

    Returns
    -------
    np.ndarray, shape (B,)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    weight_vec = np.array([
        -weights[m] if m == "mean_surprisal" else weights[m]
        for m in METRIC_ORDER
    ], dtype=np.float32)

    delta = states_after - states_before        # (B, 6)
    rewards = (delta * weight_vec).sum(axis=1)  # (B,)

    # Regression penalty
    ciu_idx = METRIC_ORDER.index("ciu_rate")
    mc_idx = METRIC_ORDER.index("mc_score")
    penalty_mask = (delta[:, ciu_idx] < 0) | (delta[:, mc_idx] < 0)
    rewards[penalty_mask] += REGRESSION_PENALTY

    return np.clip(rewards, clip[0], clip[1]).astype(np.float32)



# Internal helpers

def _validate_weights(weights: Dict[str, float]) -> None:
    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-4):
        raise ValueError(
            f"Metric weights must sum to 1.0, got {total:.4f}. "
            f"Adjust weights in configs/default.yaml."
        )
    for metric in METRIC_ORDER:
        if metric not in weights:
            raise KeyError(f"Weight missing for metric '{metric}'.")
