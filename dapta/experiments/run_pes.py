"""
Phase 2a: Patient Environment Simulator (PES) Training.

What this script does:
  1. Loads DAE outputs (state vectors, longitudinal participants)
  2. Builds (s, a, s') transition triples from longitudinal sessions
  3. Augments with RCT-derived effect size priors for single-session patients
  4. Trains the neural transition model (MLP with MC Dropout)
  5. Clusters patients into 6 groups for personalised RL training
  6. Builds TherapyEnv instances for each patient
  7. Saves transition model, cluster assignments, and environments

Requires:
  outputs/dae/  (from run_dae.py)

Outputs (used by run_rl.py):
  outputs/pes/transition_model.pt     - trained MLP weights
  outputs/pes/cluster_assignments.json
  outputs/pes/env_initial_states.npz  - initial state per patient
  outputs/pes/pes_report.json         - training stats

Usage:
  python experiments/run_pes.py
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dapta.pes.transition_model import TransitionModel
from dapta.pes.cluster import PatientClusterer
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.dae.state_builder import PatientProfile
from dapta.prta.action_space import N_ACTIONS
from dapta.utils.logger import get_logger
from dapta.utils.config import Config

logger = get_logger(__name__, log_file="logs/run_pes.log")


# ---------------------------------------------------------------------------
# RCT-derived effect size priors per exercise × discourse metric
# Based on Gorshkov et al. (2025) and Upton et al. (2024)
# Shape: (N_ACTIONS, 6) — 6 discourse metrics
# ---------------------------------------------------------------------------
RCT_PRIORS = np.array([
    # CIU   MC    MLU   TTR   SynComp Surprisal(-)
    [0.04, 0.02, 0.01, 0.02, 0.01, -0.02],   # 0: SFA naming
    [0.03, 0.01, 0.02, 0.02, 0.01, -0.01],   # 1: Phonological cueing
    [0.03, 0.02, 0.05, 0.02, 0.03, -0.02],   # 2: Sentence SVO
    [0.03, 0.02, 0.06, 0.02, 0.05, -0.02],   # 3: Sentence complex syntax
    [0.07, 0.05, 0.03, 0.03, 0.02, -0.04],   # 4: CILT dialogue
    [0.06, 0.06, 0.04, 0.03, 0.03, -0.03],   # 5: Script training
    [0.06, 0.07, 0.04, 0.04, 0.03, -0.03],   # 6: Story retelling
    [0.08, 0.06, 0.04, 0.04, 0.02, -0.04],   # 7: Conversation partner
    [0.02, 0.01, 0.01, 0.02, 0.01, -0.01],   # 8: Reading comprehension
    [0.02, 0.01, 0.01, 0.01, 0.01, -0.01],   # 9: Writing to dictation
    [0.02, 0.01, 0.01, 0.01, 0.00, -0.01],   # 10: Word-picture matching
    [0.09, 0.07, 0.05, 0.05, 0.03, -0.05],   # 11: Free conversation
], dtype=np.float32)


def build_transition_triples_from_longitudinal(
    state_vectors: np.ndarray,
    session_ids: np.ndarray,
    longitudinal: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (s, a, s') triples from participants with 2+ sessions.

    Since AphasiaBank doesn't have explicit exercise logs, we infer
    the action from session-to-session changes using cosine similarity
    to the RCT prior effect vectors — the action whose prior best
    matches the observed delta is assigned as the inferred action.

    Returns
    -------
    states      : (N, STATE_DIM)
    actions     : (N,) int
    next_states : (N, STATE_DIM)
    """
    session_to_vec = {
        sid: state_vectors[i]
        for i, sid in enumerate(session_ids)
    }

    states_list, actions_list, next_states_list = [], [], []

    for base_id, session_list in longitudinal.items():
        for i in range(len(session_list) - 1):
            s_id = session_list[i].lower()
            ns_id = session_list[i + 1].lower()

            # Find matching session vectors (case-insensitive)
            s_vec = None
            ns_vec = None
            for sid, vec in session_to_vec.items():
                if sid.lower() == s_id:
                    s_vec = vec
                if sid.lower() == ns_id:
                    ns_vec = vec

            if s_vec is None or ns_vec is None:
                continue

            delta = ns_vec[:6] - s_vec[:6]  # discourse dims only

            # Infer action: match delta to RCT prior
            prior_discourse = RCT_PRIORS[:, :6]
            similarities = prior_discourse @ delta
            inferred_action = int(np.argmax(similarities))

            states_list.append(s_vec)
            actions_list.append(inferred_action)
            next_states_list.append(ns_vec)

    if not states_list:
        logger.warning("No longitudinal triples found. Using prior-only data.")
        return np.zeros((0, 14)), np.zeros(0, dtype=int), np.zeros((0, 14))

    logger.info(f"Built {len(states_list)} transition triples from longitudinal data.")
    return (
        np.stack(states_list),
        np.array(actions_list, dtype=int),
        np.stack(next_states_list),
    )


def augment_with_priors(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    all_initial_states: np.ndarray,
    n_augment: int = 500,
    noise_std: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment longitudinal triples with RCT prior-based synthetic triples.
    This addresses the small sample size limitation of AphasiaBank.
    """
    rng = np.random.default_rng(42)
    aug_states, aug_actions, aug_next = [], [], []

    for _ in range(n_augment):
        # Sample a random initial state
        idx = rng.integers(0, len(all_initial_states))
        s = all_initial_states[idx].copy()

        # Sample a random action
        a = rng.integers(0, N_ACTIONS)

        # Apply RCT prior + noise
        delta = np.zeros(14, dtype=np.float32)
        delta[:6] = RCT_PRIORS[a] + rng.normal(0, noise_std, 6)
        ns = np.clip(s + delta, 0.0, 1.0)

        aug_states.append(s)
        aug_actions.append(a)
        aug_next.append(ns)

    aug_s = np.stack(aug_states)
    aug_a = np.array(aug_actions, dtype=int)
    aug_ns = np.stack(aug_next)

    if len(states) > 0:
        combined_s = np.concatenate([states, aug_s])
        combined_a = np.concatenate([actions, aug_a])
        combined_ns = np.concatenate([next_states, aug_ns])
    else:
        combined_s, combined_a, combined_ns = aug_s, aug_a, aug_ns

    logger.info(
        f"Augmented: {len(states)} real + {n_augment} synthetic = {len(combined_s)} total triples"
    )
    return combined_s, combined_a, combined_ns


def main(args) -> None:
    cfg = Config.load()
    output_dir = Path("outputs/pes")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 2a: Patient Environment Simulator")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load DAE outputs
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading DAE outputs...")
    dae_dir = Path("outputs/dae")

    if not (dae_dir / "state_vectors.npz").exists():
        logger.error("DAE outputs not found. Run experiments/run_dae.py first.")
        return

    dae_data = np.load(str(dae_dir / "state_vectors.npz"), allow_pickle=True)
    state_vectors = dae_data["state_vectors"]
    session_ids = dae_data["session_ids"]

    with open(dae_dir / "patient_profiles.json") as f:
        profiles_data = json.load(f)

    with open(dae_dir / "longitudinal.json") as f:
        longitudinal = json.load(f)

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    logger.info(f"Loaded {len(state_vectors)} state vectors, {len(longitudinal)} longitudinal patients")

    # Build PatientProfile objects
    profiles = [
        PatientProfile(
            participant_id=p["participant_id"],
            aphasia_subtype=p.get("aphasia_subtype", "Other"),
            wab_aq=p.get("wab_aq", 50.0),
            months_post_onset=p.get("months_post_onset", 12.0),
        )
        for p in profiles_data
    ]

    # ------------------------------------------------------------------
    # Build transition triples
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Building transition triples from longitudinal data...")
    states, actions, next_states = build_transition_triples_from_longitudinal(
        state_vectors, session_ids, longitudinal
    )

    # Augment with RCT priors
    states, actions, next_states = augment_with_priors(
        states, actions, next_states,
        all_initial_states=state_vectors,
        n_augment=args.n_augment,
    )

    # ------------------------------------------------------------------
    # Train transition model
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Training transition model (MLP + MC Dropout)...")
    transition_model = TransitionModel(
        hidden_sizes=(128, 64),
        dropout=0.2,
        mc_samples=20,
        device=args.device,
        checkpoint_path="outputs/pes/transition_model.pt",
    )

    transition_model.fit(
        states=states,
        actions=actions,
        next_states=next_states,
        val_split=0.1,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=args.epochs,
        early_stopping_patience=15,
    )

    logger.info("Transition model trained.")

    # ------------------------------------------------------------------
    # Cluster patients
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Clustering patients...")
    clusterer = PatientClusterer(n_clusters=8, random_seed=42)
    cluster_labels = clusterer.fit_predict(profiles)
    cluster_groups = clusterer.get_cluster_groups(profiles, cluster_labels)

    cluster_assignments = {
        p.participant_id: int(label)
        for p, label in zip(profiles, cluster_labels)
    }

    for cluster_id, group in cluster_groups.items():
        logger.info(f"  Cluster {cluster_id}: {len(group)} patients")

    # ------------------------------------------------------------------
    # Build environments
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Building TherapyEnv instances...")

    # Initial state per session = state vector
    initial_states = state_vectors

    envs = build_env_population(
        initial_states=list(initial_states),
        transition_model=transition_model,
        episode_horizon=20,
    )

    # Group envs by cluster
    session_to_cluster = {}
    for i, (sid, profile) in enumerate(zip(session_ids, profiles)):
        session_to_cluster[sid] = int(cluster_labels[i])

    cluster_env_indices: Dict[int, List[int]] = {i: [] for i in range(8)}
    for i, sid in enumerate(session_ids):
        c = session_to_cluster.get(sid, 0)
        cluster_env_indices[c].append(i)

    # Save initial states for environments
    np.savez(
        str(output_dir / "env_initial_states.npz"),
        initial_states=initial_states,
        session_ids=session_ids,
        cluster_labels=cluster_labels,
    )

    # Save cluster assignments
    with open(output_dir / "cluster_assignments.json", "w") as f:
        json.dump({
            "assignments": cluster_assignments,
            "cluster_env_indices": {str(k): v for k, v in cluster_env_indices.items()},
            "cluster_sizes": {str(k): len(v) for k, v in cluster_env_indices.items()},
        }, f, indent=2)

    # Save report
    report = {
        "n_transition_triples": len(states),
        "n_longitudinal_real": int((states == states).all(axis=1).sum()),
        "n_augmented_synthetic": args.n_augment,
        "n_clusters": 8,
        "cluster_sizes": {str(k): len(v) for k, v in cluster_env_indices.items()},
        "n_environments": len(envs),
        "train_split_size": len(splits["train"]),
        "val_split_size": len(splits["val"]),
        "test_split_size": len(splits["test"]),
    }
    with open(output_dir / "pes_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2a Complete.")
    logger.info(f"  Transition triples : {len(states)}")
    logger.info(f"  Patient clusters   : 6")
    logger.info(f"  Environments built : {len(envs)}")
    logger.info(f"  Outputs saved to   : {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_rl.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 2a: PES Training")
    parser.add_argument(
        "--n_augment", type=int, default=500,
        help="Number of synthetic RCT-prior triples to augment training data (default: 500)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs for transition model (default: 100)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (auto-detected if not set)"
    )
    args = parser.parse_args()
    main(args)
