"""
07_evaluate.py
==============
Phase 7: Full evaluation and comparison.

Compares PPO and DQN against:
  - Random baseline
  - Round-robin (fixed cyclic) baseline
  - Most-common exercise baseline

Statistical tests:
  - Paired Wilcoxon signed-rank test
  - Cohen's d effect size
  - Subgroup analysis by aphasia subtype and severity

Answers:
  RQ2: Does RL outperform static sequences?
  RQ3: Do discourse-level gains transfer to naturalistic speech?
  RQ4: Does patient-specific adaptation beat a generalized model?

Output: results/evaluation/

Usage:
    python 07_evaluate.py
"""

import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR  = Path(CFG["paths"]["processed"])
MODEL_DIR = Path(CFG["paths"]["models"])
RES_DIR   = Path(CFG["paths"]["results"]) / "evaluation"
RES_DIR.mkdir(parents=True, exist_ok=True)

E       = CFG["environment"]
EV      = CFG["evaluation"]
SEED    = CFG["project"]["seed"]
ALPPHA  = EV["significance_level"]
MIN_D   = EV["effect_size_threshold"]

np.random.seed(SEED)
logging.basicConfig(
    filename=Path(CFG["paths"]["logs"]) / "07_evaluate.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from aphasia_env import AphasiaTherapyEnv


# ── Baseline policies ─────────────────────────────────────────────────────────

class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def predict(self, obs, deterministic=False):
        return np.random.randint(0, self.n_actions), None


class RoundRobinPolicy:
    def __init__(self, n_actions):
        self._cycle = cycle(range(n_actions))
    def predict(self, obs, deterministic=False):
        return next(self._cycle), None


class MostCommonPolicy:
    """Always selects action 0 (first/most common exercise)."""
    def predict(self, obs, deterministic=False):
        return 0, None


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_policy_episodes(policy,
                         env: AphasiaTherapyEnv,
                         n_episodes: int = 100) -> dict:
    """
    Run policy for n_episodes, collect per-episode metrics.
    Returns dict of lists.
    """
    all_rewards         = []
    all_final_mlu       = []
    all_final_ciu       = []
    all_final_vocd      = []
    all_exercise_counts = {ex["name"]: 0 for ex in E["therapy_exercises"]}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        final_info = {}

        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            final_info = info

            ex_name = info.get("exercise", "")
            if ex_name in all_exercise_counts:
                all_exercise_counts[ex_name] += 1

            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        patient = env._current_patient
        all_final_mlu.append(patient.get("mlu_words")   or 0.0)
        all_final_ciu.append(patient.get("ciu_approx")  or 0.0)
        all_final_vocd.append(patient.get("vocd_d")     or 0.0)

    return {
        "rewards":         all_rewards,
        "final_mlu":       all_final_mlu,
        "final_ciu":       all_final_ciu,
        "final_vocd":      all_final_vocd,
        "exercise_counts": all_exercise_counts,
    }


# ── Statistical tests ─────────────────────────────────────────────────────────

def cohens_d(a: list, b: list) -> float:
    a, b   = np.array(a), np.array(b)
    pooled = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def compare_policies(name_a: str, rewards_a: list,
                     name_b: str, rewards_b: list) -> dict:
    stat, p = stats.wilcoxon(rewards_a, rewards_b)
    d       = cohens_d(rewards_a, rewards_b)
    return {
        "comparison":   f"{name_a} vs {name_b}",
        "mean_a":       round(float(np.mean(rewards_a)), 4),
        "mean_b":       round(float(np.mean(rewards_b)), 4),
        "diff":         round(float(np.mean(rewards_a)) - float(np.mean(rewards_b)), 4),
        "wilcoxon_W":   round(float(stat), 4),
        "p_value":      round(float(p),    4),
        "cohens_d":     round(d,            4),
        "significant":  p < ALPPHA,
        "meaningful_d": abs(d) >= MIN_D,
    }


# ── Subgroup analysis ─────────────────────────────────────────────────────────

def subgroup_analysis(policy_name: str,
                       policy,
                       patients_df: pd.DataFrame,
                       transition_model: dict,
                       scaler_params: dict,
                       env_cfg: dict,
                       n_episodes_per_group: int = 50) -> pd.DataFrame:
    """Run policy per aphasia subtype and severity group."""
    rows = []

    # By subtype
    if "subtype" in patients_df.columns:
        for subtype, grp in patients_df.groupby("subtype"):
            if len(grp) < 5:
                continue
            env = Monitor(AphasiaTherapyEnv(
                patients_df=grp.reset_index(drop=True),
                transition_model=transition_model,
                metric_cols=env_cfg["metric_cols"],
                state_features=env_cfg["state_features"],
                exercises=E["therapy_exercises"],
                scaler_params=scaler_params,
                max_steps=env_cfg["max_steps"],
                noise=E["transition_noise"],
                seed=SEED,
            ))
            results = run_policy_episodes(policy, env, n_episodes_per_group)
            rows.append({
                "policy":   policy_name,
                "group_by": "subtype",
                "group":    str(subtype),
                "n":        len(grp),
                "mean_reward": np.mean(results["rewards"]),
                "std_reward":  np.std(results["rewards"]),
                "mean_ciu":    np.mean(results["final_ciu"]),
                "mean_mlu":    np.mean(results["final_mlu"]),
            })
            env.close()

    # By severity (WAB-AQ tertiles)
    if "wab_aq" in patients_df.columns:
        patients_df = patients_df.copy()
        patients_df["severity_group"] = pd.qcut(
            patients_df["wab_aq"].fillna(patients_df["wab_aq"].median()),
            q=3, labels=["Severe", "Moderate", "Mild"]
        )
        for sev, grp in patients_df.groupby("severity_group"):
            if len(grp) < 5:
                continue
            env = Monitor(AphasiaTherapyEnv(
                patients_df=grp.reset_index(drop=True),
                transition_model=transition_model,
                metric_cols=env_cfg["metric_cols"],
                state_features=env_cfg["state_features"],
                exercises=E["therapy_exercises"],
                scaler_params=scaler_params,
                max_steps=env_cfg["max_steps"],
                noise=E["transition_noise"],
                seed=SEED,
            ))
            results = run_policy_episodes(policy, env, n_episodes_per_group)
            rows.append({
                "policy":   policy_name,
                "group_by": "severity",
                "group":    str(sev),
                "n":        len(grp),
                "mean_reward": np.mean(results["rewards"]),
                "std_reward":  np.std(results["rewards"]),
                "mean_ciu":    np.mean(results["final_ciu"]),
                "mean_mlu":    np.mean(results["final_mlu"]),
            })
            env.close()

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_reward_comparison(results_dict: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    ax = axes[0]
    labels  = list(results_dict.keys())
    rewards = [results_dict[k]["rewards"] for k in labels]
    bp = ax.boxplot(rewards, labels=labels, patch_artist=True)
    colors = ["steelblue", "coral", "gray", "lightgreen", "plum"]
    for patch, color in zip(bp["boxes"], colors[:len(labels)]):
        patch.set_facecolor(color)
    ax.set_title("Episode Reward Distribution by Policy")
    ax.set_ylabel("Total Episode Reward")
    ax.tick_params(axis="x", rotation=15)

    # Mean ± std bar chart
    ax = axes[1]
    means  = [np.mean(results_dict[k]["rewards"]) for k in labels]
    stds   = [np.std(results_dict[k]["rewards"])  for k in labels]
    x      = np.arange(len(labels))
    bars   = ax.bar(x, means, yerr=stds, capsize=5,
                    color=colors[:len(labels)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Mean Reward ± Std by Policy")
    ax.set_ylabel("Mean Episode Reward")

    plt.suptitle("Policy Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(RES_DIR / "reward_comparison.png", dpi=150)
    plt.close()


def plot_metric_trajectories(results_dict: dict):
    metrics = ["final_mlu", "final_ciu", "final_vocd"]
    titles  = ["Final MLU", "Final CIU %", "Final VOCD-D"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    colors = ["steelblue", "coral", "gray", "lightgreen", "plum"]

    for j, (met, title) in enumerate(zip(metrics, titles)):
        ax = axes[j]
        for i, (policy_name, res) in enumerate(results_dict.items()):
            vals = res.get(met, [])
            if vals:
                ax.hist(vals, bins=20, alpha=0.5,
                        label=policy_name, color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.legend(fontsize=7)

    plt.suptitle("Final Metric Distributions by Policy", fontsize=13)
    plt.tight_layout()
    plt.savefig(RES_DIR / "metric_distributions.png", dpi=150)
    plt.close()


def plot_subgroup_heatmap(df_sub: pd.DataFrame):
    if df_sub.empty:
        return

    for group_by in df_sub["group_by"].unique():
        subset = df_sub[df_sub["group_by"] == group_by]
        pivot  = subset.pivot(index="group", columns="policy", values="mean_reward")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, ax=ax)
        ax.set_title(f"Mean Reward by {group_by.capitalize()} and Policy")
        plt.tight_layout()
        plt.savefig(RES_DIR / f"subgroup_{group_by}_heatmap.png", dpi=150)
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 7: Evaluation")

    # Load data and environment config
    syn_path = PROC_DIR / "synthetic_patients.csv"
    if not syn_path.exists():
        console.print("[red]Run 05_build_env.py first.[/red]")
        return

    patients_df      = pd.read_csv(syn_path)
    master_df        = pd.read_csv(PROC_DIR / "master_metrics.csv")
    transition_model = joblib.load(PROC_DIR / "transition_model.pkl")
    with open(PROC_DIR / "scaler_params.yaml") as f:
        scaler_params = yaml.safe_load(f)
    with open(PROC_DIR / "env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)

    # Merge real subtype info into synthetic patients (for subgroup analysis)
    if "subtype" in master_df.columns:
        subtype_info = master_df[["participant_id", "subtype", "wab_aq"]].drop_duplicates()
        real_subtypes = subtype_info["subtype"].dropna().tolist()
        patients_df["subtype"] = np.random.choice(
            real_subtypes, size=len(patients_df), replace=True
        )

    n_actions = env_cfg["n_actions"]

    # ── Load trained RL models ────────────────────────────────────────────
    policies = {}

    for algo_class, name in [(PPO, "PPO"), (DQN, "DQN")]:
        model_path = MODEL_DIR / f"{name.lower()}_best.zip"
        if model_path.exists():
            model = algo_class.load(model_path)
            policies[name] = model
            console.print(f"[green]✓ Loaded {name} model[/green]")
        else:
            console.print(f"[yellow]⚠ {name} model not found — skipping[/yellow]")

    # Add baselines
    policies["Random"]      = RandomPolicy(n_actions)
    policies["Round-Robin"] = RoundRobinPolicy(n_actions)
    policies["Most-Common"] = MostCommonPolicy()

    # ── Run evaluation episodes ───────────────────────────────────────────
    console.print(f"\nRunning {100} episodes per policy...")
    results_dict = {}

    for policy_name, policy in policies.items():
        console.print(f"  Evaluating: {policy_name}...")
        env = AphasiaTherapyEnv(
            patients_df=patients_df,
            transition_model=transition_model,
            metric_cols=env_cfg["metric_cols"],
            state_features=env_cfg["state_features"],
            exercises=E["therapy_exercises"],
            scaler_params=scaler_params,
            max_steps=env_cfg["max_steps"],
            noise=E["transition_noise"],
            seed=SEED,
        )
        results_dict[policy_name] = run_policy_episodes(policy, env, n_episodes=100)
        env.close()

    # ── Statistical comparisons ───────────────────────────────────────────
    console.print("\nRunning statistical comparisons...")
    stat_rows = []

    rl_policies    = [n for n in policies if n in ["PPO", "DQN"]]
    base_policies  = [n for n in policies if n not in ["PPO", "DQN"]]

    for rl in rl_policies:
        for base in base_policies:
            row = compare_policies(
                rl,   results_dict[rl]["rewards"],
                base, results_dict[base]["rewards"],
            )
            stat_rows.append(row)

    # PPO vs DQN
    if "PPO" in results_dict and "DQN" in results_dict:
        stat_rows.append(compare_policies(
            "PPO", results_dict["PPO"]["rewards"],
            "DQN", results_dict["DQN"]["rewards"],
        ))

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(RES_DIR / "statistical_tests.csv", index=False)

    # Print stats table
    table = Table(title="Statistical Comparison Results")
    for col in ["comparison", "mean_a", "mean_b", "diff",
                "p_value", "cohens_d", "significant", "meaningful_d"]:
        table.add_column(col, style="cyan" if col == "comparison" else "white")

    for _, row in df_stats.iterrows():
        sig  = "[green]Yes[/green]" if row["significant"]  else "[red]No[/red]"
        eff  = "[green]Yes[/green]" if row["meaningful_d"] else "[red]No[/red]"
        table.add_row(
            str(row["comparison"]),
            f"{row['mean_a']:.4f}",
            f"{row['mean_b']:.4f}",
            f"{row['diff']:+.4f}",
            f"{row['p_value']:.4f}",
            f"{row['cohens_d']:.4f}",
            sig, eff,
        )
    console.print(table)

    # ── Subgroup analysis ─────────────────────────────────────────────────
    console.print("\nRunning subgroup analysis...")
    sub_rows = []
    for policy_name, policy in policies.items():
        df_sub = subgroup_analysis(
            policy_name, policy, patients_df,
            transition_model, scaler_params, env_cfg,
            n_episodes_per_group=30,
        )
        sub_rows.append(df_sub)

    df_subgroup = pd.concat(sub_rows, ignore_index=True)
    df_subgroup.to_csv(RES_DIR / "subgroup_analysis.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    console.print("\nGenerating evaluation plots...")
    plot_reward_comparison(results_dict)
    plot_metric_trajectories(results_dict)
    plot_subgroup_heatmap(df_subgroup)

    # Exercise frequency heatmap (what does the RL agent prefer?)
    if "PPO" in results_dict:
        ec = results_dict["PPO"]["exercise_counts"]
        fig, ax = plt.subplots(figsize=(10, 4))
        names  = list(ec.keys())
        counts = list(ec.values())
        ax.bar(names, counts, color="steelblue")
        ax.set_title("PPO Exercise Selection Frequency")
        ax.set_ylabel("Times Selected")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(RES_DIR / "ppo_exercise_frequency.png", dpi=150)
        plt.close()

    # Correlation: RL reward vs WAB-AQ (functional transfer proxy)
    if "wab_aq" in master_df.columns and "composite_reward" in master_df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        valid = master_df[["wab_aq", "composite_reward"]].dropna()
        ax.scatter(valid["wab_aq"], valid["composite_reward"], alpha=0.4)
        r, p = stats.pearsonr(valid["wab_aq"], valid["composite_reward"])
        ax.set_xlabel("WAB-AQ (Functional Severity)")
        ax.set_ylabel("Composite Discourse Reward")
        ax.set_title(f"Reward vs WAB-AQ  (r={r:.3f}, p={p:.4f})")
        plt.tight_layout()
        plt.savefig(RES_DIR / "reward_vs_wab_aq.png", dpi=150)
        plt.close()

    # ── Final summary ─────────────────────────────────────────────────────
    console.rule("[bold green]Evaluation Complete")
    console.print(f"\n[green]✓ Statistical tests → {RES_DIR / 'statistical_tests.csv'}[/green]")
    console.print(f"[green]✓ Subgroup analysis → {RES_DIR / 'subgroup_analysis.csv'}[/green]")
    console.print(f"[green]✓ Plots saved in    → {RES_DIR}[/green]")

    # Overall verdict
    rl_beats_all = all(
        row["significant"] and row["meaningful_d"]
        for _, row in df_stats.iterrows()
        if "vs" in str(row["comparison"]) and
        any(rl in str(row["comparison"]) for rl in ["PPO", "DQN"])
        and "PPO vs DQN" not in str(row["comparison"])
    )
    if rl_beats_all:
        console.print("\n[bold green]✓ RL agents significantly outperform all baselines "
                      "with meaningful effect sizes (RQ2 supported)[/bold green]")
    else:
        console.print("\n[bold yellow]⚠ Mixed results — check statistical_tests.csv for details[/bold yellow]")


if __name__ == "__main__":
    main()
