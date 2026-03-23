# does this have different enironments for test, val, test? 
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dapta.pes.transition_model import TransitionModel
from dapta.pes.cluster import PatientClusterer
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.dae.state_builder import PatientProfile, STATE_DIM, N_TASKS, N_METRICS_PER_TASK
from dapta.prta.action_space import N_ACTIONS
from dapta.utils.logger import get_logger
from dapta.utils.config import Config

logger = get_logger(__name__, log_file="logs/run_pes.log")

RCT_PRIORS = np.array([
    [0.04, 0.02, 0.01, 0.02, 0.01, -0.02],   # 0:  SFA naming
    [0.03, 0.01, 0.02, 0.02, 0.01, -0.01],   # 1:  Phonological cueing
    [0.03, 0.02, 0.05, 0.02, 0.03, -0.02],   # 2:  Sentence SVO
    [0.03, 0.02, 0.06, 0.02, 0.05, -0.02],   # 3:  Sentence complex syntax
    [0.07, 0.05, 0.03, 0.03, 0.02, -0.04],   # 4:  CILT dialogue
    [0.06, 0.06, 0.04, 0.03, 0.03, -0.03],   # 5:  Script training
    [0.06, 0.07, 0.04, 0.04, 0.03, -0.03],   # 6:  Story retelling
    [0.08, 0.06, 0.04, 0.04, 0.02, -0.04],   # 7:  Conversation partner
    [0.02, 0.01, 0.01, 0.02, 0.01, -0.01],   # 8:  Reading comprehension
    [0.02, 0.01, 0.01, 0.01, 0.01, -0.01],   # 9:  Writing to dictation
    [0.02, 0.01, 0.01, 0.01, 0.00, -0.01],   # 10: Word-picture matching
    [0.09, 0.07, 0.05, 0.05, 0.03, -0.05],   # 11: Free conversation
], dtype=np.float32)

METRIC_NAMES_VAL = [
    "ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"
]


# ======================================================================
def build_transition_triples_from_longitudinal(
    state_vectors: np.ndarray,
    session_ids:   np.ndarray,
    longitudinal:  Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    session_to_vec = {
        sid: state_vectors[i]
        for i, sid in enumerate(session_ids)
    }

    states_list, actions_list, next_states_list = [], [], []

    for base_id, session_list in longitudinal.items():
        for i in range(len(session_list) - 1):
            s_id  = session_list[i].lower()
            ns_id = session_list[i + 1].lower()

            s_vec = ns_vec = None
            for sid, vec in session_to_vec.items():
                if sid.lower() == s_id:
                    s_vec = vec
                if sid.lower() == ns_id:
                    ns_vec = vec

            if s_vec is None or ns_vec is None:
                continue

            delta           = ns_vec[:5] - s_vec[:5]
            prior_discourse = RCT_PRIORS[:, :5]
            similarities    = prior_discourse @ delta
            inferred_action = int(np.argmax(similarities))

            states_list.append(s_vec)
            actions_list.append(inferred_action)
            next_states_list.append(ns_vec)

    if not states_list:
        logger.warning("No longitudinal triples found. Using prior-only data.")
        return (
            np.zeros((0, STATE_DIM), dtype=np.float32),
            np.zeros(0, dtype=int),
            np.zeros((0, STATE_DIM), dtype=np.float32),
        )

    logger.info(f"Built {len(states_list)} transition triples from longitudinal data.")
    return (
        np.stack(states_list),
        np.array(actions_list, dtype=int),
        np.stack(next_states_list),
    )


# ======================================================================
def augment_with_priors(
    states:             np.ndarray,
    actions:            np.ndarray,
    next_states:        np.ndarray,
    all_initial_states: np.ndarray,
    n_augment:          int   = 500,
    noise_std:          float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(42)
    aug_states, aug_actions, aug_next = [], [], []

    for _ in range(n_augment):
        idx       = rng.integers(0, len(all_initial_states))
        s         = all_initial_states[idx].copy()
        a         = rng.integers(0, N_ACTIONS)
        delta     = np.zeros(STATE_DIM, dtype=np.float32)
        for t in range(N_TASKS):
            base = t * N_METRICS_PER_TASK
            delta[base:base + 6] = RCT_PRIORS[a] + rng.normal(0, noise_std, 6)
        ns        = np.clip(s + delta, 0.0, 1.0)
        aug_states.append(s)
        aug_actions.append(a)
        aug_next.append(ns)

    aug_s  = np.stack(aug_states)
    aug_a  = np.array(aug_actions, dtype=int)
    aug_ns = np.stack(aug_next)

    if len(states) > 0:
        combined_s  = np.concatenate([states,      aug_s])
        combined_a  = np.concatenate([actions,     aug_a])
        combined_ns = np.concatenate([next_states, aug_ns])
    else:
        combined_s, combined_a, combined_ns = aug_s, aug_a, aug_ns

    logger.info(
        f"Augmented: {len(states)} real + {n_augment} synthetic = "
        f"{len(combined_s)} total triples"
    )
    return combined_s, combined_a, combined_ns


# ======================================================================
def main(args) -> None:
    cfg        = Config.load()
    output_dir = Path("outputs/pes")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 2a: Patient Environment Simulator")
    logger.info("=" * 60)

    # ── 1. Load DAE outputs ──────────────────────────────────────────
    logger.info("\n[1/5] Loading DAE outputs...")
    dae_dir = Path("outputs/dae")

    if not (dae_dir / "state_vectors.npz").exists():
        logger.error("DAE outputs not found. Run experiments/run_dae.py first.")
        return

    dae_data      = np.load(str(dae_dir / "state_vectors.npz"), allow_pickle=True)
    state_vectors = dae_data["state_vectors"]
    session_ids   = dae_data["session_ids"]

    with open(dae_dir / "patient_profiles.json") as f:
        profiles_data = json.load(f)

    with open(dae_dir / "longitudinal.json") as f:
        longitudinal = json.load(f)

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    logger.info(
        f"Loaded {len(state_vectors)} state vectors, "
        f"{len(longitudinal)} longitudinal patients"
    )

    # Build PatientProfile objects
    profiles = [
        PatientProfile(
            participant_id  = p["participant_id"],
            aphasia_subtype = p.get("aphasia_subtype", "Other"),
            wab_aq          = p.get("wab_aq", 50.0),
        )
        for p in profiles_data
    ]

    # Build session → participant and session → split lookup tables
    session_to_participant = {
        p["session_id"]: p["participant_id"]
        for p in profiles_data
    }

    # Split labels: map each participant_id to its split name
    participant_to_split: Dict[str, str] = {}
    for split_name, id_list in splits.items():
        for pid in id_list:
            participant_to_split[pid] = split_name

    # Assign a split label to every session
    split_labels = np.array([
        participant_to_split.get(
            session_to_participant.get(str(sid), ""), "train"
        )
        for sid in session_ids
    ])

    train_mask = split_labels == "train"
    val_mask   = split_labels == "val"
    test_mask  = split_labels == "test"

    logger.info(
        f"Session split: {train_mask.sum()} train, "
        f"{val_mask.sum()} val, "
        f"{test_mask.sum()} test"
    )

    # ── 2. Build transition triples (train only) ─────────────────────
    logger.info("\n[2/5] Building transition triples from longitudinal data...")

    # Longitudinal triples are inherently train-only because longitudinal
    # participants are assigned to train in run_dae.py splits.
    states, actions, next_states = build_transition_triples_from_longitudinal(
        state_vectors, session_ids, longitudinal
    )
    n_real     = len(states)
    real_split = int(n_real * 0.9)

    val_states_real  = states[real_split:]
    val_actions_real = actions[real_split:]
    val_next_real    = next_states[real_split:]
    train_states     = states[:real_split]
    train_actions    = actions[:real_split]
    train_next       = next_states[:real_split]

    # Augment using train patient states only to avoid leakage
    train_initial_states = state_vectors[train_mask]
    train_states, train_actions, train_next = augment_with_priors(
        train_states, train_actions, train_next,
        all_initial_states = train_initial_states,
        n_augment          = args.n_augment,
    )

    # ── 3. Train transition model ────────────────────────────────────
    logger.info("\n[3/5] Training transition model (MLP + MC Dropout)...")
    transition_model = TransitionModel(
        hidden_sizes    = (128, 64),
        dropout         = 0.2,
        mc_samples      = 20,
        device          = args.device,
        checkpoint_path = "outputs/pes/transition_model.pt",
    )

    transition_model.fit(
        states                  = train_states,
        actions                 = train_actions,
        next_states             = train_next,
        val_split               = 0.1,
        learning_rate           = 1e-3,
        batch_size              = 32,
        num_epochs              = args.epochs,
        early_stopping_patience = 15,
    )
    logger.info("Transition model trained.")

    # ── 3b. Validate transition model ───────────────────────────────
    logger.info("\n[3b/5] Validating transition model...")

    val_metrics_out = {
        "val_mse":              None,
        "mae_per_metric":       {},
        "directional_accuracy": {},
        "note":                 None,
    }

    if len(val_states_real) == 0:
        logger.warning("No real longitudinal validation data. Skipping.")
        val_metrics_out["note"] = (
            "Skipped: no held-out real longitudinal triples available."
        )
    else:
        val_deltas  = val_next_real - val_states_real
        pred_nexts  = np.array([
            transition_model.predict_next_state(s, int(a))
            for s, a in zip(val_states_real, val_actions_real)
        ])
        pred_deltas = pred_nexts - val_states_real
        mae_per_dim = np.mean(
            np.abs(pred_deltas[:, :5] - val_deltas[:, :5]), axis=0
        )
        mse_overall = float(np.mean((pred_deltas - val_deltas) ** 2))
        dir_acc     = np.mean(
            np.sign(pred_deltas[:, :5]) == np.sign(val_deltas[:, :5]), axis=0
        )

        logger.info(f"  Val MSE (overall): {mse_overall:.6f}")
        for i, m in enumerate(METRIC_NAMES_VAL):
            logger.info(
                f"  {m:<25}  MAE={mae_per_dim[i]:.4f}  DirAcc={dir_acc[i]:.3f}"
            )

        val_metrics_out["val_mse"] = round(mse_overall, 6)
        val_metrics_out["mae_per_metric"] = {
            m: round(float(mae_per_dim[i]), 4)
            for i, m in enumerate(METRIC_NAMES_VAL)
        }
        val_metrics_out["directional_accuracy"] = {
            m: round(float(dir_acc[i]), 3)
            for i, m in enumerate(METRIC_NAMES_VAL)
        }

    with open(output_dir / "transition_model_validation.json", "w") as f:
        json.dump(val_metrics_out, f, indent=2)
    logger.info("  Saved to outputs/pes/transition_model_validation.json")

    # ── 4. Cluster patients (aphasia only) ───────────────────────────
    logger.info("\n[4/5] Clustering patients...")

    raw_subtypes   = [p.get("aphasia_subtype", "Other") for p in profiles_data]
    clusterer      = PatientClusterer(max_k=10, random_seed=42)
    cluster_labels = clusterer.fit_predict(
        profiles,
        state_vectors = state_vectors,
        min_k         = 6,
        raw_subtypes  = raw_subtypes,
    )
    cluster_groups = clusterer.get_cluster_groups(profiles, cluster_labels)
    import pickle
    with open(output_dir / "clusterer.pkl", "wb") as f:
        pickle.dump(clusterer, f)
    n_clusters     = clusterer.n_clusters

    cluster_assignments = {
        p.participant_id: int(label)
        for p, label in zip(profiles, cluster_labels)
    }

    for cluster_id, group in cluster_groups.items():
        logger.info(f"  Cluster {cluster_id}: {len(group)} patients")
        logger.info(
            f"    Sample subtypes: {[p.aphasia_subtype for p in group[:5]]}"
        )

    # ── 5. Build TherapyEnv instances for ALL patients ───────────────
    logger.info("\n[5/5] Building TherapyEnv instances...")

    # Build environments for every patient (train + val + test).
    # The split_labels array saved below lets run_rl.py filter correctly:
    #   - RL agent trains only on train environments
    #   - Evaluation runs on test environments
    # This way the transition model is trained once and shared across splits.
    envs = build_env_population(
        initial_states   = list(state_vectors),
        transition_model = transition_model,
        episode_horizon  = 20,
    )

    session_to_cluster = {
        sid: int(cluster_labels[i])
        for i, sid in enumerate(session_ids)
    }

    # Per-cluster environment indices — train patients only for RL training
    cluster_env_indices: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    for i, sid in enumerate(session_ids):
        c = session_to_cluster.get(sid, -1)
        if c >= 0 and split_labels[i] == "train":   # train only
            cluster_env_indices[c].append(i)

    # ── Save outputs ─────────────────────────────────────────────────
    np.savez(
        str(output_dir / "env_initial_states.npz"),
        initial_states = state_vectors,
        session_ids    = session_ids,
        cluster_labels = cluster_labels,
        split_labels   = split_labels,   # "train" / "val" / "test" per session
    )

    with open(output_dir / "cluster_assignments.json", "w") as f:
        json.dump(
            {
                "assignments":         cluster_assignments,
                "cluster_env_indices": {
                    str(k): v for k, v in cluster_env_indices.items()
                },
                "cluster_sizes": {
                    str(k): len(v) for k, v in cluster_env_indices.items()
                },
            },
            f, indent=2,
        )
    
    report = {
        "n_transition_triples":  len(train_states),
        "n_longitudinal_real":   real_split,
        "n_augmented_synthetic": args.n_augment,
        "n_clusters":            n_clusters,
        "cluster_sizes": {
            str(k): len(v) for k, v in cluster_env_indices.items()
        },
        "n_environments_total":  len(envs),
        "n_environments_train":  int(train_mask.sum()),
        "n_environments_val":    int(val_mask.sum()),
        "n_environments_test":   int(test_mask.sum()),
        "train_split_size":      len(splits["train"]),
        "val_split_size":        len(splits["val"]),
        "test_split_size":       len(splits["test"]),
    }
    with open(output_dir / "pes_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2a Complete.")
    logger.info(f"  Transition triples      : {len(train_states)}")
    logger.info(f"  Patient clusters        : {n_clusters}")
    logger.info(f"  Environments (total)    : {len(envs)}")
    logger.info(f"  Environments (train)    : {train_mask.sum()}")
    logger.info(f"  Environments (val)      : {val_mask.sum()}")
    logger.info(f"  Environments (test)     : {test_mask.sum()}")
    logger.info(f"  Outputs saved to        : {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_rl.py")


# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 2a: PES Training")
    parser.add_argument(
        "--n_augment", type=int, default=500,
        help="Synthetic RCT-prior triples to augment training data (default: 500)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Max training epochs for transition model (default: 100)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (auto-detected if not set)",
    )
    args = parser.parse_args()
    main(args)