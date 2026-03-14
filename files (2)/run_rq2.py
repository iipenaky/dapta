"""
run_rq2.py
==========
Answers RQ2: Can a reinforcement learning agent learn patient-specific
therapy exercise sequences that outperform static or rule-based sequences
in improving discourse-level outcomes?

What this script does
---------------------
  1.  Loads state_vectors.csv (from run_rq1.py + cluster.py)
  2.  Builds the transition dataset (real + RCT-prior augmented)
  3.  Trains the TransitionModel (neural patient simulator)
  4.  Runs Optuna HPO — jointly optimises:
        - Reward weights (5 logits -> softmax)
        - DDQN hyperparameters (architecture, lr, gamma, etc.)
      Objective: maximise mean cumulative discourse improvement
      on held-out patients after N training episodes.
  5.  Trains the final G-DDQN agent with best found params
  6.  Evaluates three static baselines:
        - Greedy       : always pick highest generalisation_potential
        - Round-robin  : cycle through all 12 exercises
        - Random       : uniform random action selection
  7.  Statistical comparison: Wilcoxon signed-rank test
      G-DDQN vs each baseline (non-parametric, paired, n<30)
  8.  Saves all results, plots, and best hyperparameters

Outputs
-------
  models/gddqn.pt                      trained G-DDQN checkpoint
  models/transition_model.pt           trained transition model
  data/processed/transitions.csv       (s,a,s') training triples
  results/rq2/best_params.json         best HPO hyperparameters
  results/rq2/best_weights.json        best reward weights
  results/rq2/gddqn_vs_baselines.csv   comparison table
  results/rq2/training_curve.png       G-DDQN learning curve
  results/rq2/action_distribution.png  what exercises the agent picks
  results/rq2/per_metric_improvement.png  improvement per discourse metric

Run
---
  python run_rq2.py

Flags
-----
  --state_vectors PATH   default: data/processed/state_vectors.csv
  --out_dir       PATH   default: results/rq2
  --n_trials      INT    Optuna trials (default: 50)
  --n_episodes    INT    training episodes per trial (default: 500)
  --final_episodes INT   episodes for final G-DDQN training (default: 2000)
  --eval_episodes INT    evaluation episodes (default: 50)
  --train_split   FLOAT  fraction of patients for training (default: 0.8)
  --seed          INT    random seed (default: 42)
  --skip_hpo             load best_params.json instead of re-running HPO
"""

import argparse
import json
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from reward           import (
    METRIC_ORDER, N_METRICS,
    softmax_weights, weights_from_trial, weights_to_dict, validate_weights,
)
from transition_model import TransitionModel
from transitions      import build_transition_dataset, save_transitions
from therapy_env      import TherapyEnv, build_env_population
from ddqn             import (
    DDQNAgent,
    GreedyBaseline, RoundRobinBaseline, RandomBaseline,
    hyperparams_from_trial, PPOAgent
)
from action_space     import ACTION_ID_TO_NAME, N_ACTIONS


# =============================================================================
# DEFAULTS
# =============================================================================

_STATE_VECTORS   = "data/processed/state_vectors.csv"
_TRANSITIONS_CSV = "data/processed/transitions.csv"
_GDDQN_CKPT      = "models/gddqn.pt"
_TM_CKPT         = "models/transition_model.pt"
_OUT_DIR         = "results/rq2"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_split(
    state_vectors_csv: str,
    train_split:       float,
    seed:              int,
) -> tuple:
    """
    Load state vectors, split aphasia patients into train/eval sets.

    Controls (cluster_id == -1) are excluded entirely.
    Split is stratified by cluster_id so both sets have proportional
    representation of each patient group.

    Returns
    -------
    df_train, df_eval, df_full (aphasia only)
    """
    df = pd.read_csv(state_vectors_csv)
    df["mean_surprisal"] = df["mean_surprisal"].fillna(0.5)
    required = ["participant_id", "cluster_id"] + METRIC_ORDER
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"state_vectors.csv missing columns: {missing}\n"
            f"Run run_rq1.py then cluster.py first."
        )

    # Exclude controls
    df_aphasia = df[df["cluster_id"] >= 0].copy().reset_index(drop=True)
    print(f"  {len(df_aphasia)} aphasia patients available for RL training")

    # Stratified split by cluster
    rng        = np.random.default_rng(seed)
    train_rows, eval_rows = [], []

    for cid, group in df_aphasia.groupby("cluster_id"):
        idx     = group.index.tolist()
        rng.shuffle(idx)
        split   = max(1, int(len(idx) * train_split))
        train_rows.extend(idx[:split])
        eval_rows.extend(idx[split:])

    df_train = df_aphasia.loc[train_rows].reset_index(drop=True)
    df_eval  = df_aphasia.loc[eval_rows ].reset_index(drop=True)

    print(f"  Train: {len(df_train)} patients | Eval: {len(df_eval)} patients")
    return df_train, df_eval, df_aphasia


def df_to_states(df: pd.DataFrame) -> list:
    """Convert DataFrame rows to list of normalised state vectors."""
    states = []
    for _, row in df.iterrows():
        s = row[METRIC_ORDER].values.astype(np.float32)
        s = np.nan_to_num(s, nan=0.5)
        s = np.clip(s, 0.0, 1.0)
        states.append(s)
    return states


# =============================================================================
# ENVIRONMENT BUILDER
# =============================================================================

def build_envs(
    df:               pd.DataFrame,
    transition_model: TransitionModel,
    weights:          np.ndarray,
) -> list:
    """Build one TherapyEnv per patient row with given reward weights."""
    return build_env_population(
        initial_states   = df_to_states(df),
        transition_model = transition_model,
        reward_weights   = weights,
    )


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def make_objective(
    df_train:         pd.DataFrame,
    df_eval:          pd.DataFrame,
    transition_model: TransitionModel,
    n_episodes:       int,
    eval_episodes:    int,
    seed:             int,
):
    """
    Returns an Optuna objective function that jointly optimises
    reward weights and DDQN hyperparameters.

    The objective:
      1. Samples reward weights via softmax (sum=1, all positive)
      2. Samples DDQN hyperparameters
      3. Builds train/eval environments with those weights
      4. Trains a DDQN agent for n_episodes
      5. Returns mean cumulative improvement on eval set

    Both the reward definition and agent configuration are optimised
    together because they interact — the best architecture depends on
    the reward signal shape, which depends on the weights.
    """
    def objective(trial):
        # Sample reward weights
        weights = weights_from_trial(trial)

        # Sample DDQN hyperparameters
        hp = hyperparams_from_trial(trial)

        # Build environments with these weights
        train_envs = build_envs(df_train, transition_model, weights)
        eval_envs  = build_envs(df_eval,  transition_model, weights)

        # Train agent
        agent = DDQNAgent(
            hidden_sizes        = hp["hidden_sizes"],
            dropout             = hp["dropout"],
            learning_rate       = hp["learning_rate"],
            gamma               = hp["gamma"],
            epsilon_decay_steps = hp["epsilon_decay_steps"],
            batch_size          = hp["batch_size"],
            buffer_capacity     = hp["buffer_capacity"],
            target_update_freq  = hp["target_update_freq"],
            seed                = seed,
        )
        agent.train(
            envs          = train_envs,
            n_episodes    = n_episodes,
            eval_envs     = eval_envs,
            eval_every    = max(10, n_episodes // 20),
            eval_episodes = eval_episodes,
        )

        # Objective: mean cumulative improvement on eval patients
        score = agent.evaluate(eval_envs, n_episodes=eval_episodes)

        # Log weights for inspection
        trial.set_user_attr("weights", weights_to_dict(weights))
        trial.set_user_attr("hyperparams", hp)

        return score

    return objective


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curve(
    history: dict,
    out_path: str,
) -> None:
    episode_rewards = history["episode_rewards"]
    eval_scores     = history["eval_scores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Smoothed training rewards
    window = max(1, len(episode_rewards) // 50)
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    ax1.plot(episode_rewards, alpha=0.3, color="steelblue", lw=0.8)
    ax1.plot(smoothed, color="steelblue", lw=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("G-DDQN training reward")

    # Eval scores
    if eval_scores:
        xs, ys = zip(*eval_scores)
        ax2.plot(xs, ys, "o-", color="coral", lw=2, markersize=5)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Mean cumulative improvement")
        ax2.set_title("G-DDQN eval score (held-out patients)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curve -> {out_path}")


def plot_action_distribution(
    results: dict,
    out_path: str,
) -> None:
    """Bar chart comparing action distributions across agent and baselines."""
    from action_space import THERAPY_EXERCISES

    fig, axes = plt.subplots(
        1, len(results), figsize=(5 * len(results), 6), sharey=True
    )
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        dist   = res.get("action_distribution", {})
        labels = [ex.name for ex in THERAPY_EXERCISES]
        values = [dist.get(label, 0.0) for label in labels]
        colors = [
            "#2196F3" if ex.target_level == "discourse" else
            "#FF9800" if ex.target_level == "sentence"  else
            "#9E9E9E"
            for ex in THERAPY_EXERCISES
        ]
        ax.barh(labels, values, color=colors)
        ax.set_title(name)
        ax.set_xlabel("Mean selections per episode")

    plt.suptitle(
        "Exercise selection — G-DDQN vs baselines\n"
        "Blue=discourse, Orange=sentence, Grey=word",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Action distribution -> {out_path}")


def plot_per_metric_improvement(
    results: dict,
    out_path: str,
) -> None:
    """Grouped bar chart: per-metric improvement for each agent/baseline."""
    agents  = list(results.keys())
    metrics = METRIC_ORDER
    x       = np.arange(len(metrics))
    width   = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["per_metric"].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Mean improvement (normalised)")
    ax.set_title("Per-metric discourse improvement — G-DDQN vs baselines")
    ax.legend()
    ax.axhline(0, color="black", lw=0.8, ls="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Per-metric improvement -> {out_path}")


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def statistical_comparison(
    agent_scores:    list,
    baseline_scores: dict,
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test: G-DDQN vs each baseline.

    Non-parametric paired test appropriate for small samples.
    Reports W statistic, p-value, and effect size r = Z/sqrt(N).

    Parameters
    ----------
    agent_scores    : list of per-episode total improvement for G-DDQN
    baseline_scores : dict name -> list of per-episode total improvement

    Returns
    -------
    pd.DataFrame with one row per baseline
    """
    rows = []
    for name, scores in baseline_scores.items():
        n     = min(len(agent_scores), len(scores))
        a     = np.array(agent_scores[:n])
        b     = np.array(scores[:n])
        try:
            stat, p = wilcoxon(a, b, alternative="greater")
            z       = stat  # approximate
            r       = z / np.sqrt(n)
        except Exception:
            stat, p, r = float("nan"), float("nan"), float("nan")

        rows.append({
            "baseline":           name,
            "G-DDQN_mean":        float(np.mean(a)),
            "baseline_mean":      float(np.mean(b)),
            "mean_diff":          float(np.mean(a) - np.mean(b)),
            "wilcoxon_W":         float(stat),
            "p_value":            float(p),
            "effect_size_r":      float(r),
            "significant_p05":    p < 0.05 if not np.isnan(p) else False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAPTA RQ2 — G-DDQN vs static baselines"
    )
    parser.add_argument("--state_vectors",   default=_STATE_VECTORS)
    parser.add_argument("--out_dir",         default=_OUT_DIR)
    parser.add_argument("--n_trials",        default=50,   type=int)
    parser.add_argument("--n_episodes",      default=500,  type=int,
                        help="Training episodes per Optuna trial")
    parser.add_argument("--final_episodes",  default=2000, type=int,
                        help="Episodes for final G-DDQN training")
    parser.add_argument("--eval_episodes",   default=50,   type=int)
    parser.add_argument("--train_split",     default=0.8,  type=float)
    parser.add_argument("--seed",            default=42,   type=int)
    parser.add_argument("--skip_hpo",        action="store_true",
                        help="Load saved best_params.json instead of HPO")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(_GDDQN_CKPT).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DAPTA -- RQ2: G-DDQN vs Static Baselines")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Step 1: Load + split data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading and splitting patient data...")
    df_train, df_eval, df_aphasia = load_and_split(
        args.state_vectors, args.train_split, args.seed
    )

    # ------------------------------------------------------------------
    # Step 2: Build transition dataset + train TransitionModel
    # ------------------------------------------------------------------
    print("\n[Step 2] Building transition dataset...")
    transition_model = TransitionModel(checkpoint_path=_TM_CKPT)

    states, actions, next_states = build_transition_dataset(
        df               = df_aphasia,
        transition_model = transition_model,
        augment          = True,
    )
    save_transitions(states, actions, next_states, _TRANSITIONS_CSV)

    print("\n[Step 3] Training TransitionModel...")
    transition_model.fit(
        states      = states,
        actions     = actions,
        next_states = next_states,
    )

    # ------------------------------------------------------------------
    # Step 3: HPO — jointly learn reward weights + agent hyperparams
    # ------------------------------------------------------------------
    best_params_path = out_dir / "best_params.json"

    if args.skip_hpo and best_params_path.exists():
        print("\n[Step 4] Loading saved HPO results...")
        with open(best_params_path) as f:
            saved = json.load(f)
        best_weights = np.array(
            list(saved["weights"].values()), dtype=np.float32
        )
        best_hp = saved["hyperparams"]
        print(f"  Loaded weights: {saved['weights']}")
        print(f"  Loaded hyperparams: {best_hp}")

    else:
        print(f"\n[Step 4] Running Optuna HPO ({args.n_trials} trials)...")
        print(f"  Optimising reward weights + DDQN hyperparameters jointly.")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("pip install optuna")

        objective = make_objective(
            df_train         = df_train,
            df_eval          = df_eval,
            transition_model = transition_model,
            n_episodes       = args.n_episodes,
            eval_episodes    = args.eval_episodes,
            seed             = args.seed,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)

        best_trial   = study.best_trial
        best_weights = np.array(
            list(best_trial.user_attrs["weights"].values()),
            dtype=np.float32,
        )
        best_hp = best_trial.user_attrs["hyperparams"]

        print(f"\n  Best HPO score: {study.best_value:.4f}")
        print(f"  Best weights:   {weights_to_dict(best_weights)}")
        print(f"  Best agent hp:  {best_hp}")

        # Save best params
        saved = {
            "best_value":  study.best_value,
            "weights":     weights_to_dict(best_weights),
            "hyperparams": best_hp,
        }
        with open(best_params_path, "w") as f:
            json.dump(saved, f, indent=2)
        print(f"  Best params -> {best_params_path}")

        # Save weights separately for use in run_rq4.py
        with open(out_dir / "best_weights.json", "w") as f:
            json.dump(weights_to_dict(best_weights), f, indent=2)

    # ------------------------------------------------------------------
    # Step 4: Train final G-DDQN with best params on full training set
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Training final G-DDQN "
          f"({args.final_episodes} episodes)...")

    train_envs = build_envs(df_train, transition_model, best_weights)
    eval_envs  = build_envs(df_eval,  transition_model, best_weights)

    gddqn = DDQNAgent(
        **best_hp,
        seed             = args.seed,
        checkpoint_path  = _GDDQN_CKPT,
    )
    history = gddqn.train(
        envs          = train_envs,
        n_episodes    = args.final_episodes,
        eval_envs     = eval_envs,
        eval_every    = 100,
        eval_episodes = args.eval_episodes,
    )

    plot_training_curve(history, str(out_dir / "training_curve.png"))

        # ------------------------------------------------------------------
    # Step 5b: Train PPO agent with same best params
    # ------------------------------------------------------------------
    print(f"\n[Step 5b] Training PPO agent ({args.final_episodes} episodes)...")

    ppo_agent = PPOAgent(
        lr              = best_hp.get("learning_rate", 3e-4),
        gamma           = best_hp.get("gamma", 0.95),
        hidden_sizes    = best_hp.get("hidden_sizes", [128, 64]),
        seed            = args.seed,
        checkpoint_path = "models/ppo.pt",
    )
    ppo_history = ppo_agent.train(
        envs          = train_envs,
        n_episodes    = args.final_episodes,
        eval_envs     = eval_envs,
        eval_every    = 100,
        eval_episodes = args.eval_episodes,
    )
    print(f"  PPO best eval: {ppo_agent._best_reward:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Evaluate G-DDQN + all three baselines
    # ------------------------------------------------------------------
    print("\n[Step 6] Evaluating G-DDQN and static baselines...")

    # Collect per-episode scores for statistical test
    all_envs = build_envs(
        pd.concat([df_train, df_eval]).reset_index(drop=True),
        transition_model,
        best_weights,
    )

    gddqn_detailed  = gddqn.evaluate_detailed(
        all_envs, n_episodes=args.eval_episodes
    )
    greedy_detailed  = GreedyBaseline().evaluate(
        all_envs, n_episodes=args.eval_episodes
    )
    rr_detailed      = RoundRobinBaseline().evaluate(
        all_envs, n_episodes=args.eval_episodes
    )
    random_detailed  = RandomBaseline(seed=args.seed).evaluate(
        all_envs, n_episodes=args.eval_episodes
    )
    ppo_detailed = ppo_agent.evaluate_detailed(
        all_envs, n_episodes=args.eval_episodes
    )

    all_results = {
        "G-DDQN":      gddqn_detailed,
        "PPO":         ppo_detailed,
        "Greedy":      greedy_detailed,
        "Round-robin": rr_detailed,
        "Random":      random_detailed,
    }

    # Print summary table
    print(f"\n{'Agent/Baseline':<16} {'Mean improvement':>18} {'Std':>8}")
    print("-" * 46)
    for name, res in all_results.items():
        print(
            f"  {name:<14} "
            f"  {res['mean_improvement']:>16.4f} "
            f"  {res['std_improvement']:>6.4f}"
        )

    # ------------------------------------------------------------------
    # Step 6: Statistical tests
    # ------------------------------------------------------------------
    print("\n[Step 7] Statistical comparison (Wilcoxon signed-rank)...")

    agent_ep_scores = [
        p["total"] for p in gddqn_detailed["per_patient"]
    ]
    baseline_ep_scores = {
        "PPO":         [p["total"] for p in ppo_detailed.get("per_patient", [{"total": ppo_detailed["mean_improvement"]}] * args.eval_episodes)],
        "Greedy":      [p["total"] for p in greedy_detailed.get("per_patient", [{"total": greedy_detailed["mean_improvement"]}] * args.eval_episodes)],
        "Round-robin": [p["total"] for p in rr_detailed.get("per_patient",     [{"total": rr_detailed["mean_improvement"]}]     * args.eval_episodes)],
        "Random":      [p["total"] for p in random_detailed.get("per_patient", [{"total": random_detailed["mean_improvement"]}] * args.eval_episodes)],
    }

    stats_df = statistical_comparison(agent_ep_scores, baseline_ep_scores)
    stats_path = out_dir / "gddqn_vs_baselines.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n{stats_df.to_string(index=False)}")
    print(f"\n  Stats table -> {stats_path}")

    # ------------------------------------------------------------------
    # Step 7: Plots
    # ------------------------------------------------------------------
    print("\n[Step 8] Generating plots...")
    plot_action_distribution(
        all_results, str(out_dir / "action_distribution.png")
    )
    plot_per_metric_improvement(
        all_results, str(out_dir / "per_metric_improvement.png")
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  RQ2 complete.")
    print(f"  G-DDQN mean improvement : "
          f"{gddqn_detailed['mean_improvement']:.4f}")
    print(f"  PPO mean improvement    : "
      f"{ppo_detailed['mean_improvement']:.4f}")
    print(f"  Greedy baseline         : "
          f"{greedy_detailed['mean_improvement']:.4f}")
    print(f"  Round-robin baseline    : "
          f"{rr_detailed['mean_improvement']:.4f}")
    print(f"  Random baseline         : "
          f"{random_detailed['mean_improvement']:.4f}")
    print(f"\n  Best weights -> {out_dir / 'best_weights.json'}")
    print(f"  G-DDQN model -> {_GDDQN_CKPT}")
    print(f"{'='*65}")
    print("\nNext step: run run_rq3.py for transfer analysis.")


if __name__ == "__main__":
    main()