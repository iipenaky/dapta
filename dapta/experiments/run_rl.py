import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats

from dapta.pes.transition_model import TransitionModel
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import (
    train_ddqn,
    train_ppo,
    RuleBasedBaseline,
    RandomBaseline,
)
from dapta.dae.state_builder import STATE_DIM
from dapta.utils.logger import get_logger
from dapta.utils.config import Config

logger = get_logger(__name__, log_file="logs/run_rl.log")

METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
SURPRISAL_DIM      = 38
CLINICAL_THRESHOLD = 0.40

# Environment helpers
def load_environments(
    transition_model: TransitionModel,
    initial_states:   np.ndarray,
    cluster_labels:   np.ndarray,
    episode_horizon:  int = 20,
) -> Dict[int, List[TherapyEnv]]:
    n_clusters   = int(cluster_labels.max()) + 1
    cluster_envs = {}
    for cluster_id in range(n_clusters):
        mask           = cluster_labels == cluster_id
        cluster_states = initial_states[mask]
        if len(cluster_states) == 0:
            logger.warning(f"Cluster {cluster_id} has no patients.")
            cluster_envs[cluster_id] = []
            continue
        envs = build_env_population(
            initial_states   = list(cluster_states),
            transition_model = transition_model,
            episode_horizon  = episode_horizon,
        )
        cluster_envs[cluster_id] = envs
        logger.info(f"  Cluster {cluster_id}: {len(envs)} environments")
    return cluster_envs



# Agent factories
def make_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    return DDQNAgent(
        state_dim      = STATE_DIM,
        gru_hidden     = 128,
        gru_layers     = 2,
        learning_rate  = 1e-4,
        gamma          = 0.95,
        tau            = 0.005,
        buffer_size    = 10_000,
        batch_size     = 64,
        eps_start      = 1.0,
        eps_end        = 0.05,
        eps_fraction   = 0.3,
        history_len    = 10,
        use_gru        = True,
        checkpoint_path = checkpoint_path,
    )


def make_no_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    return DDQNAgent(
        state_dim      = STATE_DIM,
        gru_hidden     = 128,
        gru_layers     = 2,
        learning_rate  = 1e-4,
        gamma          = 0.95,
        tau            = 0.005,
        buffer_size    = 10_000,
        batch_size     = 64,
        eps_start      = 1.0,
        eps_end        = 0.05,
        eps_fraction   = 0.3,
        history_len    = 10,
        use_gru        = False,
        checkpoint_path = checkpoint_path,
    )


class PPOWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def select_action(self, state: np.ndarray) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)



# Episode runner
def run_episodes(
    agent,
    envs:    List[TherapyEnv],
    is_ddqn: bool = True,
    greedy:  bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    disc_imps = []
    surp_imps = []
    rewards   = []

    for env in envs:
        state, _       = env.reset()
        state_history  = []
        action_history = []
        total_reward   = 0.0
        initial_surp   = float(state[SURPRISAL_DIM])

        for _ in range(env.episode_horizon):
            if is_ddqn:
                history_tensor = agent.build_history_tensor(
                    state_history, action_history
                )
                action = agent.select_action(
                    state, history_tensor, greedy=greedy
                )
            else:
                action = agent.select_action(state)

            state_history.append(state.copy())
            action_history.append(action)

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        disc_imp   = env.get_cumulative_discourse_improvement()[:5]
        final_surp = float(state[SURPRISAL_DIM])

        disc_imps.append(disc_imp)
        surp_imps.append(initial_surp - final_surp)
        rewards.append(total_reward)

    return (
        np.stack(disc_imps),
        np.array(surp_imps, dtype=np.float32),
        np.array(rewards,   dtype=np.float32),
    )


# Main


def main(args) -> None:
    cfg        = Config.load()
    output_dir = Path("outputs/rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 2b: RL Agent Training : Full Ablation")
    logger.info("=" * 60)

    #  1. Load PES outputs 
    logger.info("\n[1/6] Loading PES outputs...")
    pes_dir = Path("outputs/pes")
    if not (pes_dir / "transition_model.pt").exists():
        logger.error("PES outputs not found. Run experiments/run_pes.py first.")
        return

    pes_data       = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    initial_states = pes_data["initial_states"]
    cluster_labels = pes_data["cluster_labels"]
    split_labels   = pes_data["split_labels"]   # "train" / "val" / "test"

    with open("outputs/dae/patient_profiles.json") as f:
        profiles_data = json.load(f)

    transition_model = TransitionModel(
        hidden_sizes    = (128, 64),
        dropout         = 0.2,
        checkpoint_path = str(pes_dir / "transition_model.pt"),
        device          = args.device,
    )
    transition_model.load()
    logger.info("Transition model loaded.")

    #  2. Filter to TRAIN patients only 
    # This is the data leakage fix.
    # Agents are trained exclusively on train environments.
    # Val and test patients are held out and evaluated in run_evaluation.py.
    logger.info("\n[2/6] Filtering to train-only environments...")

    train_mask            = split_labels == "train"
    train_initial_states  = initial_states[train_mask]
    train_cluster_labels  = cluster_labels[train_mask]
    train_profiles        = [
        p for p, m in zip(profiles_data, train_mask) if m
    ]

    logger.info(
        f"  Train: {train_mask.sum()}  "
        f"Val: {(split_labels == 'val').sum()}  "
        f"Test: {(split_labels == 'test').sum()}  "
        f"(val and test excluded from RL training)"
    )

    cluster_envs = load_environments(
        transition_model = transition_model,
        initial_states   = train_initial_states,
        cluster_labels   = train_cluster_labels,
        episode_horizon  = 20,
    )
    all_envs   = [env for envs in cluster_envs.values() for env in envs]
    n_clusters = len(cluster_envs)
    logger.info(f"  Train environments: {len(all_envs)}, clusters: {n_clusters}")

    #  3. Train DDQN agents (GRU + No-GRU) ─
    logger.info(f"\n[3/6] Training DDQN agents ({args.total_steps} steps each)...")
    training_logs: Dict = {}

    # 3a : DAPTA: GRU-DDQN cluster-specific (personalised)
    logger.info("\n  3a. DAPTA : GRU-DDQN cluster-specific")
    cluster_agents: Dict[int, DDQNAgent] = {}
    for cluster_id, envs in cluster_envs.items():
        if not envs:
            continue
        ckpt  = str(output_dir / f"ddqn_cluster_{cluster_id}.pt")
        agent = make_gru_ddqn(checkpoint_path=ckpt)
        logs  = train_ddqn(
            agent           = agent,
            envs            = envs,
            total_steps     = args.total_steps,
            eval_freq       = max(100, args.total_steps // 50),
            n_eval_episodes = min(10, len(envs)),
        )
        agent.save()
        cluster_agents[cluster_id] = agent
        training_logs[f"dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    Cluster {cluster_id} saved.")

    # 3b : G-DDQN: GRU-DDQN generalised
    logger.info("\n  3b. G-DDQN : GRU-DDQN generalised")
    g_ddqn = make_gru_ddqn(checkpoint_path=str(output_dir / "ddqn_generalised.pt"))
    training_logs["g_ddqn"] = train_ddqn(
        agent           = g_ddqn,
        envs            = all_envs,
        total_steps     = args.total_steps,
        eval_freq       = max(100, args.total_steps // 50),
        n_eval_episodes = min(20, len(all_envs)),
    )
    g_ddqn.save()
    logger.info("    G-DDQN saved.")

    # 3c : No-GRU DAPTA: No-GRU DDQN cluster-specific
    logger.info("\n  3c. No-GRU DAPTA : No-GRU DDQN cluster-specific")
    no_gru_cluster_agents: Dict[int, DDQNAgent] = {}
    for cluster_id, envs in cluster_envs.items():
        if not envs:
            continue
        ckpt  = str(output_dir / f"no_gru_ddqn_cluster_{cluster_id}.pt")
        agent = make_no_gru_ddqn(checkpoint_path=ckpt)
        logs  = train_ddqn(
            agent           = agent,
            envs            = envs,
            total_steps     = args.total_steps,
            eval_freq       = max(100, args.total_steps // 50),
            n_eval_episodes = min(10, len(envs)),
        )
        agent.save()
        no_gru_cluster_agents[cluster_id] = agent
        training_logs[f"no_gru_dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    No-GRU cluster {cluster_id} saved.")

    # 3d : No-GRU G-DDQN: No-GRU DDQN generalised
    logger.info("\n  3d. No-GRU G-DDQN : No-GRU DDQN generalised")
    no_gru_g_ddqn = make_no_gru_ddqn(
        checkpoint_path = str(output_dir / "no_gru_ddqn_generalised.pt")
    )
    training_logs["no_gru_g_ddqn"] = train_ddqn(
        agent           = no_gru_g_ddqn,
        envs            = all_envs,
        total_steps     = args.total_steps,
        eval_freq       = max(100, args.total_steps // 50),
        n_eval_episodes = min(20, len(all_envs)),
    )
    no_gru_g_ddqn.save()
    logger.info("    No-GRU G-DDQN saved.")

    #  4. Train PPO agents 
    logger.info(f"\n[4/6] Training PPO agents...")

    ppo_generalised  = None
    ppo_cluster_models = {}

    if not args.skip_ppo:
        logger.info("\n  4a. PPO generalised : pooled train environments")
        ppo_save       = str(output_dir / "ppo_generalised")
        ppo_generalised = train_ppo(
            envs         = all_envs,
            total_steps = args.total_steps,
            save_path   = ppo_save,
        )
        training_logs["ppo_generalised"] = {
            "status": "trained" if ppo_generalised else "skipped_no_sb3",
            "steps":  args.total_steps,
        }

        logger.info("\n  4b. PPO personalised : cluster-specific train environments")
        for cluster_id, envs in cluster_envs.items():
            if not envs:
                continue
            ppo_c_save  = str(output_dir / f"ppo_cluster_{cluster_id}")
            ppo_cluster = train_ppo(
                envs         = envs,
                total_steps = args.total_steps,
                save_path   = ppo_c_save,
            )
            if ppo_cluster:
                ppo_cluster_models[cluster_id] = ppo_cluster
                training_logs[f"ppo_cluster_{cluster_id}"] = {
                    "status": "trained",
                    "steps":  args.total_steps,
                }
    else:
        logger.info("  Skipping PPO (--skip_ppo flag set).")

    #  5. Collect improvement arrays (training environments only) 
    # Note: these results are on the TRAINING set and are used only to
    # compute interim training statistics and save checkpoints.
    # The authoritative evaluation on the held-out test set is performed
    # by run_evaluation.py using split_labels to load test environments.
    logger.info("\n[5/6] Collecting train-set improvement arrays...")

    # DAPTA
    dapta_disc_list, dapta_surp_list, dapta_label_list = [], [], []
    for cluster_id, agent in cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        dapta_disc_list.append(disc)
        dapta_surp_list.append(surp)
        dapta_label_list.extend([cluster_id] * len(envs))
    dapta_disc   = np.concatenate(dapta_disc_list,  axis=0) if dapta_disc_list  else np.zeros((0, 5))
    dapta_surp   = np.concatenate(dapta_surp_list,  axis=0) if dapta_surp_list  else np.zeros(0)
    dapta_labels = np.array(dapta_label_list, dtype=int)

    # G-DDQN
    g_ddqn.eps = 0.0
    g_ddqn_disc, g_ddqn_surp, _ = run_episodes(g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # No-GRU DAPTA
    no_gru_dapta_disc_list, no_gru_dapta_surp_list = [], []
    for cluster_id, agent in no_gru_cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        no_gru_dapta_disc_list.append(disc)
        no_gru_dapta_surp_list.append(surp)
    no_gru_dapta_disc = (
        np.concatenate(no_gru_dapta_disc_list, axis=0)
        if no_gru_dapta_disc_list else np.zeros((0, 5))
    )

    # No-GRU G-DDQN
    no_gru_g_ddqn.eps = 0.0
    no_gru_g_ddqn_disc, _, _ = run_episodes(no_gru_g_ddqn, all_envs, is_ddqn=True, greedy=True)

    ppo_gen_disc = np.zeros((0, 5))
    if ppo_generalised:
        ppo_gen_wrapper = PPOWrapper(ppo_generalised)

        ppo_gen_disc, _, _ = run_episodes(
            ppo_gen_wrapper,
            all_envs,
            is_ddqn=False,
            greedy=True
        )

    # PPO personalised
    ppo_pers_disc = np.zeros((0, 5))
    if ppo_cluster_models:
        ppo_pers_disc_list = []

        for cluster_id, ppo_model in ppo_cluster_models.items():
            envs = cluster_envs[cluster_id]
            if not envs:
                continue

            ppo_pers_wrapper = PPOWrapper(ppo_model)

            disc, _, _ = run_episodes(
                ppo_pers_wrapper,
                envs,
                is_ddqn=False,
                greedy=True
            )

            ppo_pers_disc_list.append(disc)

        if ppo_pers_disc_list:
            ppo_pers_disc = np.concatenate(ppo_pers_disc_list, axis=0)

    # Baselines (on train environments)
    rbde = RuleBasedBaseline()
    rbde_disc_list, rbde_surp_list = [], []
    for env in all_envs:
        state, _ = env.reset()
        rbde.reset()
        initial_surp = float(state[SURPRISAL_DIM])
        for _ in range(env.episode_horizon):
            action = rbde.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rbde_disc_list.append(env.get_cumulative_discourse_improvement()[:5])
        rbde_surp_list.append(initial_surp - float(state[SURPRISAL_DIM]))
    rbde_disc = np.stack(rbde_disc_list)
    rbde_surp = np.array(rbde_surp_list, dtype=np.float32)

    rts = RandomBaseline()
    rts_disc_list = []
    for env in all_envs:
        state, _ = env.reset()
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rts_disc_list.append(env.get_cumulative_discourse_improvement()[:5])
    rts_disc = np.stack(rts_disc_list)

    # Save training-set improvement arrays
    np.save(str(output_dir / "train_improvements_DAPTA.npy"),         dapta_disc)
    np.save(str(output_dir / "train_improvements_G_DDQN.npy"),        g_ddqn_disc)
    np.save(str(output_dir / "train_improvements_NO_GRU_DAPTA.npy"),  no_gru_dapta_disc)
    np.save(str(output_dir / "train_improvements_NO_GRU_G_DDQN.npy"), no_gru_g_ddqn_disc)
    np.save(str(output_dir / "train_improvements_PPO_GEN.npy"),       ppo_gen_disc)
    np.save(str(output_dir / "train_improvements_PPO_PERS.npy"),      ppo_pers_disc)
    np.save(str(output_dir / "train_improvements_RBDE.npy"),          rbde_disc)
    np.save(str(output_dir / "train_improvements_RTS.npy"),           rts_disc)


    

    with open(output_dir / "training_logs.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
            if isinstance(obj, list):  return [convert(x) for x in obj]
            if isinstance(obj, dict):  return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(training_logs), f, indent=2)

    logger.info("Next step: python experiments/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPTA Phase 2b: RL Agent Training"
    )
    parser.add_argument(
        "--total_steps", type=int, default=50_000,
        help="Training steps per agent (default: 50000). Use 5000 for a quick test."
    )
    parser.add_argument(
        "--skip_ppo", action="store_true",
        help="Skip PPO training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()
    main(args)