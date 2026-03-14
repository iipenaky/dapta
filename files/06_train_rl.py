"""
06_train_rl.py
==============
Phase 6: Train RL agents with Optuna hyperparameter tuning.

Trains THREE types of models:
  1. General PPO    — trained on ALL patients combined
  2. General DQN    — trained on ALL patients combined
  3. Patient-specific PPO — fine-tuned per individual patient
     (addresses RQ4: does personalisation beat a general model?)

For RQ4: patient-specific model starts from the general model weights
         and fine-tunes on each patient's trajectory.

Output:
  models/ppo_general.zip
  models/dqn_general.zip
  models/patient_specific/ppo_{participant_id}.zip
  results/training/

Usage:
    python 06_train_rl.py
"""

import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR    = Path(CFG["paths"]["processed"])
MODEL_DIR   = Path(CFG["paths"]["models"])
PS_DIR      = MODEL_DIR / "patient_specific"
RES_DIR     = Path(CFG["paths"]["results"]) / "training"
LOG_DIR     = Path(CFG["paths"]["logs"])
for d in [MODEL_DIR, PS_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

T    = CFG["training"]
E    = CFG["environment"]
SEED = CFG["project"]["seed"]

device_cfg = CFG["project"]["device"]
DEVICE = "cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else (
    device_cfg if device_cfg != "auto" else "cpu"
)
console.print(f"Device: [bold]{DEVICE}[/bold]")

logging.basicConfig(
    filename=LOG_DIR / "06_train_rl.log", level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from aphasia_env import AphasiaTherapyEnv


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(patients_df, transition_model, scaler_params, env_cfg, rank=0):
    def _init():
        env = AphasiaTherapyEnv(
            patients_df=patients_df,
            transition_model=transition_model,
            metric_cols=env_cfg["metric_cols"],
            state_features=env_cfg["state_features"],
            exercises=E["therapy_exercises"],
            scaler_params=scaler_params,
            max_steps=env_cfg["max_steps"],
            noise=E["transition_noise"],
            seed=SEED + rank,
        )
        return Monitor(env)
    return _init


def get_policy_kwargs(arch_name: str) -> dict:
    return {"net_arch": CFG["training"]["net_arch_map"][arch_name]}


# ─────────────────────────────────────────────────────────────────────────────
# Reward logger callback
# ─────────────────────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_r = 0.0

    def _on_step(self):
        for r, d in zip(self.locals.get("rewards", []),
                        self.locals.get("dones",   [])):
            self._ep_r += r
            if d:
                self.episode_rewards.append(self._ep_r)
                self._ep_r = 0.0
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objectives
# ─────────────────────────────────────────────────────────────────────────────

def ppo_objective(trial, patients_df, transition_model, scaler_params, env_cfg):
    sp = CFG["training"]["ppo"]
    lr        = trial.suggest_float("learning_rate", *sp["learning_rate"], log=True)
    n_steps   = trial.suggest_int("n_steps",         *sp["n_steps"])
    batch     = trial.suggest_int("batch_size",      *sp["batch_size"])
    n_epochs  = trial.suggest_int("n_epochs",        *sp["n_epochs"])
    gamma     = trial.suggest_float("gamma",         *sp["gamma"])
    gae       = trial.suggest_float("gae_lambda",    *sp["gae_lambda"])
    clip      = trial.suggest_float("clip_range",    *sp["clip_range"])
    ent       = trial.suggest_float("ent_coef",      *sp["ent_coef"],    log=True)
    vf        = trial.suggest_float("vf_coef",       *sp["vf_coef"])
    arch      = trial.suggest_categorical("net_arch", sp["net_arch"])
    n_steps   = max(n_steps, batch)

    env      = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED)
    eval_env = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED+1)
    model    = PPO("MlpPolicy", env, learning_rate=lr, n_steps=n_steps,
                    batch_size=batch, n_epochs=n_epochs, gamma=gamma,
                    gae_lambda=gae, clip_range=clip, ent_coef=ent, vf_coef=vf,
                    policy_kwargs=get_policy_kwargs(arch), verbose=0, device=DEVICE, seed=SEED)
    model.learn(T["total_timesteps"] // 5)
    mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=T["n_eval_episodes"], deterministic=True)
    env.close(); eval_env.close()
    return float(mean_r)


def dqn_objective(trial, patients_df, transition_model, scaler_params, env_cfg):
    sp   = CFG["training"]["dqn"]
    lr   = trial.suggest_float("learning_rate",         *sp["learning_rate"],    log=True)
    buf  = trial.suggest_int("buffer_size",              *sp["buffer_size"])
    ls   = trial.suggest_int("learning_starts",          *sp["learning_starts"])
    bat  = trial.suggest_int("batch_size",               *sp["batch_size"])
    tau  = trial.suggest_float("tau",                    *sp["tau"],              log=True)
    gam  = trial.suggest_float("gamma",                  *sp["gamma"])
    tf   = trial.suggest_int("train_freq",               *sp["train_freq"])
    gs   = trial.suggest_int("gradient_steps",           *sp["gradient_steps"])
    tui  = trial.suggest_int("target_update_interval",   *sp["target_update_interval"])
    ef   = trial.suggest_float("exploration_fraction",   *sp["exploration_fraction"])
    ee   = trial.suggest_float("exploration_final_eps",  *sp["exploration_final_eps"])
    arch = trial.suggest_categorical("net_arch",          sp["net_arch"])

    env      = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED)
    eval_env = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED+2)
    model    = DQN("MlpPolicy", env, learning_rate=lr, buffer_size=buf,
                    learning_starts=ls, batch_size=bat, tau=tau, gamma=gam,
                    train_freq=tf, gradient_steps=gs, target_update_interval=tui,
                    exploration_fraction=ef, exploration_final_eps=ee,
                    policy_kwargs=get_policy_kwargs(arch), verbose=0, device=DEVICE, seed=SEED)
    model.learn(T["total_timesteps"] // 5)
    mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=T["n_eval_episodes"], deterministic=True)
    env.close(); eval_env.close()
    return float(mean_r)


# ─────────────────────────────────────────────────────────────────────────────
# Train general model
# ─────────────────────────────────────────────────────────────────────────────

def train_general(algo: str, best_params: dict,
                   patients_df, transition_model, scaler_params, env_cfg) -> tuple:
    """Train a general model on all patients and return (model, rewards)."""
    arch = best_params.pop("net_arch", "medium")
    pk   = get_policy_kwargs(arch)

    env      = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED)
    eval_env = make_vec_env(make_env_fn(patients_df, transition_model, scaler_params, env_cfg), n_envs=1, seed=SEED+999)

    cb_r    = RewardLogger()
    cb_eval = EvalCallback(eval_env, best_model_save_path=str(MODEL_DIR),
                            log_path=str(LOG_DIR), eval_freq=T["eval_freq"],
                            n_eval_episodes=T["n_eval_episodes"], deterministic=True, verbose=0)

    if algo == "PPO":
        bp = dict(best_params)
        bp["n_steps"] = max(bp.get("n_steps", 2048), bp.get("batch_size", 64))
        model = PPO("MlpPolicy", env, policy_kwargs=pk, verbose=0, device=DEVICE, seed=SEED, **bp)
    else:
        model = DQN("MlpPolicy", env, policy_kwargs=pk, verbose=0, device=DEVICE, seed=SEED, **best_params)

    model.learn(T["total_timesteps"], callback=[cb_r, cb_eval], progress_bar=True)
    save_path = MODEL_DIR / f"{algo.lower()}_general"
    model.save(save_path)
    console.print(f"[green]✓ General {algo} → {save_path}.zip[/green]")

    env.close(); eval_env.close()
    return model, cb_r.episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Train patient-specific models (RQ4)
# ─────────────────────────────────────────────────────────────────────────────

def train_patient_specific(general_model,
                             algo: str,
                             master_df: pd.DataFrame,
                             transition_model: dict,
                             scaler_params: dict,
                             env_cfg: dict,
                             n_finetune_steps: int = 10000,
                             min_sessions: int = 1) -> dict:
    """
    For each real participant, create an environment containing only that
    participant's data, load the general model weights, and fine-tune.

    Returns dict: {participant_id: mean_eval_reward}
    """
    results = {}
    pids    = master_df["participant_id"].unique()
    console.print(f"\nFine-tuning patient-specific models for {len(pids)} participants...")

    for pid in pids:
        patient_rows = master_df[master_df["participant_id"] == pid].copy()
        if len(patient_rows) < min_sessions:
            continue

        # Build single-patient environment
        env = Monitor(AphasiaTherapyEnv(
            patients_df=patient_rows.reset_index(drop=True),
            transition_model=transition_model,
            metric_cols=env_cfg["metric_cols"],
            state_features=env_cfg["state_features"],
            exercises=E["therapy_exercises"],
            scaler_params=scaler_params,
            max_steps=env_cfg["max_steps"],
            noise=E["transition_noise"],
            seed=SEED,
        ))

        # Load general model and fine-tune
        if algo == "PPO":
            ps_model = PPO.load(MODEL_DIR / "ppo_general.zip", env=env, device=DEVICE)
        else:
            ps_model = DQN.load(MODEL_DIR / "dqn_general.zip", env=env, device=DEVICE)

        ps_model.set_env(env)
        ps_model.learn(total_timesteps=n_finetune_steps)

        # Evaluate
        mean_r, _ = evaluate_policy(ps_model, env, n_eval_episodes=20, deterministic=True)
        results[pid] = float(mean_r)

        # Save
        ps_path = PS_DIR / f"ppo_{pid}"
        ps_model.save(ps_path)
        env.close()

        logging.info(f"Patient-specific {algo} {pid}: reward={mean_r:.4f}")

    console.print(f"[green]✓ Patient-specific models saved to {PS_DIR}[/green]")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def smooth(x, w=20):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_learning_curves(rewards_dict: dict):
    n = len(rewards_dict)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]
    colors = ["steelblue", "coral"]
    for i, (name, rewards) in enumerate(rewards_dict.items()):
        ax = axes[i]
        if rewards:
            ax.plot(rewards, alpha=0.25, color=colors[i], lw=0.7)
            s = smooth(rewards)
            ax.plot(range(len(s)), s, color=colors[i], lw=2, label="Smoothed (w=20)")
            ax.set_title(f"{name} Learning Curve")
            ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
            ax.legend()
    plt.suptitle("RL Training — General Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(RES_DIR / "learning_curves.png", dpi=150)
    plt.close()


def plot_general_vs_specific(general_rewards: dict,
                               ps_rewards_ppo: dict):
    """Compare general vs patient-specific PPO rewards per patient."""
    if not ps_rewards_ppo:
        return

    pids    = sorted(ps_rewards_ppo.keys())
    gen_r   = general_rewards.get("PPO", {})
    ps_r    = [ps_rewards_ppo.get(pid, 0) for pid in pids]
    gen_vals = [gen_r if isinstance(gen_r, float) else 0.0] * len(pids)

    fig, ax = plt.subplots(figsize=(max(10, len(pids) * 0.5), 5))
    x = np.arange(len(pids))
    ax.bar(x - 0.2, gen_vals, 0.35, label="General PPO", color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, ps_r,     0.35, label="Patient-specific PPO", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("General vs Patient-Specific PPO (RQ4)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / "general_vs_patient_specific.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 6: RL Training")

    for path, name in [(PROC_DIR / "synthetic_patients.csv",  "synthetic_patients.csv"),
                        (PROC_DIR / "transition_model.pkl",    "transition_model.pkl"),
                        (PROC_DIR / "scaler_params.yaml",      "scaler_params.yaml"),
                        (PROC_DIR / "env_config.yaml",         "env_config.yaml")]:
        if not path.exists():
            console.print(f"[red]{name} not found. Run 05_build_env.py first.[/red]")
            return

    patients_df      = pd.read_csv(PROC_DIR / "synthetic_patients.csv")
    master_df        = pd.read_csv(PROC_DIR / "master_metrics.csv")
    transition_model = joblib.load(PROC_DIR / "transition_model.pkl")
    with open(PROC_DIR / "scaler_params.yaml") as f:
        scaler_params = yaml.safe_load(f)
    with open(PROC_DIR / "env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)

    console.print(f"Training patients: {len(patients_df)} synthetic + "
                  f"{master_df['participant_id'].nunique()} real")

    rewards_dict   = {}
    general_evals  = {}
    summary_rows   = []

    # ── Train general models ──────────────────────────────────────────────
    for algo in T["algorithms"]:
        console.rule(f"[yellow]{algo} — Hyperparameter Search")
        objective_fn = ppo_objective if algo == "PPO" else dqn_objective

        study = optuna.create_study(
            direction=T["optuna"]["direction"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
            study_name=f"{algo}_general",
        )
        study.optimize(
            lambda trial: objective_fn(trial, patients_df, transition_model,
                                        scaler_params, env_cfg),
            n_trials=T["optuna"]["n_trials"],
            timeout=T["optuna"]["timeout_seconds"],
            show_progress_bar=True,
        )
        joblib.dump(study, MODEL_DIR / f"{algo.lower()}_study.pkl")

        best = dict(study.best_params)
        console.print(f"Best {algo} trial reward: {study.best_value:.4f}")

        console.rule(f"[green]Training general {algo}")
        model, ep_rewards = train_general(algo, dict(best), patients_df,
                                           transition_model, scaler_params, env_cfg)
        rewards_dict[algo] = ep_rewards

        # Final eval
        eval_env = Monitor(AphasiaTherapyEnv(
            patients_df=patients_df,
            transition_model=transition_model,
            metric_cols=env_cfg["metric_cols"],
            state_features=env_cfg["state_features"],
            exercises=E["therapy_exercises"],
            scaler_params=scaler_params,
            max_steps=env_cfg["max_steps"],
            noise=E["transition_noise"],
            seed=SEED,
        ))
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
        eval_env.close()
        general_evals[algo] = float(mean_r)

        console.print(f"General {algo} final eval: {mean_r:.4f} ± {std_r:.4f}")
        summary_rows.append({
            "model_type": "general",
            "algorithm":  algo,
            "mean_reward": mean_r,
            "std_reward":  std_r,
            **{f"hp_{k}": v for k, v in best.items()},
        })

    # ── Train patient-specific PPO (RQ4) ──────────────────────────────────
    if "PPO" in T["algorithms"]:
        ppo_general = PPO.load(MODEL_DIR / "ppo_general.zip")
        ps_results  = train_patient_specific(
            ppo_general, "PPO", master_df, transition_model,
            scaler_params, env_cfg,
            n_finetune_steps=10000,
        )

        for pid, r in ps_results.items():
            summary_rows.append({
                "model_type": "patient_specific",
                "algorithm":  "PPO",
                "participant_id": pid,
                "mean_reward": r,
            })

        # Statistical test: general vs patient-specific (RQ4)
        from scipy import stats
        gen_mean = general_evals.get("PPO", 0.0)
        ps_vals  = list(ps_results.values())
        if ps_vals:
            t_stat, p_val = stats.ttest_1samp(ps_vals, gen_mean)
            console.print(f"\n[bold]RQ4 — Patient-specific vs General PPO:[/bold]")
            console.print(f"  General mean: {gen_mean:.4f}")
            console.print(f"  PS mean:      {np.mean(ps_vals):.4f} ± {np.std(ps_vals):.4f}")
            console.print(f"  t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05 and np.mean(ps_vals) > gen_mean:
                console.print("[bold green]  ✓ Patient-specific significantly better (RQ4 supported)[/bold green]")
            else:
                console.print("[bold yellow]  ⚠ No significant difference — check results[/bold yellow]")
    else:
        ps_results = {}

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_learning_curves(rewards_dict)
    plot_general_vs_specific(general_evals, ps_results)

    # ── Save summary ──────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(RES_DIR / "training_summary.csv", index=False)

    tbl = Table(title="Training Summary")
    tbl.add_column("Type");       tbl.add_column("Algorithm")
    tbl.add_column("Mean Reward"); tbl.add_column("Std")
    for _, r in df_sum[df_sum.model_type == "general"].iterrows():
        tbl.add_row(str(r["model_type"]), str(r["algorithm"]),
                     f"{r['mean_reward']:.4f}", f"{r.get('std_reward', 0):.4f}")
    console.print(tbl)

    console.print(f"\n[green]✓ Summary → {RES_DIR / 'training_summary.csv'}[/green]")
    console.print(f"[green]✓ Learning curves → {RES_DIR / 'learning_curves.png'}[/green]")
    console.print(f"[green]✓ General vs PS plot → {RES_DIR / 'general_vs_patient_specific.png'}[/green]")


if __name__ == "__main__":
    main()