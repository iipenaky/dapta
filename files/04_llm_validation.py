"""
04_llm_validation.py
====================
Phase 4: Three-way validation against CLAN ground truth.

  A) Python metrics  vs CLAN metrics  → validates your NLP pipeline
  B) LLM scores      vs CLAN metrics  → validates LLM as a scorer
  C) Summary: which source is most reliable? → informs reward signal choice

For A: direct numeric comparison (Pearson r, MAE, Bland-Altman)
For B: LLM scores transcripts on 1-5 scales; CLAN values binned to same scale

Answers RQ1: Can discourse metrics be automatically computed with enough
accuracy to serve as a therapy reward signal?

Output: results/llm_validation/
  validation_python_vs_clan.csv
  validation_llm_vs_clan.csv
  validation_summary.csv
  bland_altman_plots.png
  scatter_plots.png

Usage:
    python 04_llm_validation.py
"""

import yaml
import json
import re
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

warnings.filterwarnings("ignore")
console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR = Path(CFG["paths"]["processed"])
RES_DIR  = Path(CFG["paths"]["results"]) / "llm_validation"
RES_DIR.mkdir(parents=True, exist_ok=True)

L = CFG["llm"]
V = L["validation"]
KAPPA_THRESH   = V["kappa_threshold"]
PEARSON_THRESH = V["pearson_threshold"]
SAMPLE_N       = V["sample_size"]

logging.basicConfig(
    filename=Path(CFG["paths"]["logs"]) / "04_llm_validation.log",
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Python vs CLAN
# ─────────────────────────────────────────────────────────────────────────────

# Map: Python metric column → CLAN metric column
PYTHON_TO_CLAN = {
    "mlu_words":  "mlu_words_clan",    # from 02_clan_metrics.py
    "ttr":        "ttr_clan",
    "ndw":        "ndw_clan",
    "vocd_d":     "vocd_d_clan",
}


def validate_python_vs_clan(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric, compute Pearson r and MAE between
    Python-computed and CLAN-computed values.
    Requires columns: mlu_words, ttr, ndw, vocd_d (Python)
                  AND mlu_words_clan, ttr_clan, ndw_clan, vocd_d_clan (CLAN)
    """
    rows = []
    for py_col, clan_col in PYTHON_TO_CLAN.items():
        if py_col not in df.columns or clan_col not in df.columns:
            console.print(f"  [yellow]Skipping {py_col} vs {clan_col} — columns not found[/yellow]")
            continue

        paired = df[[py_col, clan_col]].dropna()
        if len(paired) < 5:
            console.print(f"  [yellow]Skipping {py_col} — fewer than 5 paired rows[/yellow]")
            continue

        py_vals   = paired[py_col].values
        clan_vals = paired[clan_col].values

        r, p   = stats.pearsonr(py_vals, clan_vals)
        mae    = float(np.mean(np.abs(py_vals - clan_vals)))
        rmse   = float(np.sqrt(np.mean((py_vals - clan_vals) ** 2)))
        bias   = float(np.mean(py_vals - clan_vals))

        rows.append({
            "metric":        py_col,
            "n_pairs":       len(paired),
            "pearson_r":     round(float(r),  3),
            "pearson_p":     round(float(p),  4),
            "mae":           round(mae,        3),
            "rmse":          round(rmse,       3),
            "mean_bias":     round(bias,       3),
            "passes_r":      float(r) >= PEARSON_THRESH,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Part B: LLM vs CLAN
# ─────────────────────────────────────────────────────────────────────────────

LLM_PROMPTS = {
    "mlu_score": {
        "question": "Rate the MEAN LENGTH OF UTTERANCE. How long are the speaker's sentences on average?",
        "scale": "1=very short (1-2 words), 2=short (3-4), 3=moderate (5-7), 4=long (8-10), 5=very long (>10)",
        "clan_col": "mlu_words_clan",
        "to_ordinal": lambda v: (
            1 if v <= 2 else 2 if v <= 4 else 3 if v <= 7 else 4 if v <= 10 else 5
        ),
    },
    "lexical_diversity_score": {
        "question": "Rate LEXICAL DIVERSITY. How varied is the vocabulary?",
        "scale": "1=very repetitive, 2=low variety, 3=moderate, 4=varied, 5=very diverse",
        "clan_col": "ttr_clan",
        "to_ordinal": lambda v: (
            1 if v < 0.10 else 2 if v < 0.20 else 3 if v < 0.35 else 4 if v < 0.50 else 5
        ),
    },
    "informativeness_score": {
        "question": "Rate INFORMATIVENESS. How much relevant information is communicated?",
        "scale": "1=very little information, 2=limited, 3=moderate, 4=good, 5=rich informative content",
        "clan_col": "ndw_clan",
        "to_ordinal": lambda v: (
            1 if v < 20 else 2 if v < 50 else 3 if v < 100 else 4 if v < 200 else 5
        ),
    },
    "fluency_score": {
        "question": "Rate FLUENCY. How smooth is the speech — absence of revisions and fillers?",
        "scale": "1=very disfluent (many revisions), 2=disfluent, 3=moderate, 4=fluent, 5=very fluent",
        "clan_col": "maze_words_clan",
        "to_ordinal": lambda v: (
            5 if v == 0 else 4 if v <= 2 else 3 if v <= 5 else 2 if v <= 10 else 1
        ),
    },
}


def build_prompt(transcript: str, metric_key: str) -> str:
    mp = LLM_PROMPTS[metric_key]
    return (
        f"You are a speech-language pathologist evaluating a transcript "
        f"from a person with aphasia.\n\n"
        f"TRANSCRIPT:\n{transcript[:600]}\n\n"
        f"TASK: {mp['question']}\n"
        f"SCALE: {mp['scale']}\n\n"
        f"Respond with ONLY a JSON object, no other text:\n"
        f'{{ "score": <integer 1-5>, "reason": "<one short sentence>" }}\n\n'
        f"JSON:"
    )


def load_llm(model_name: str, device_map: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    console.print(f"Loading LLM: [bold]{model_name}[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def llm_score(prompt: str, tokenizer, model, max_new_tokens: int, temperature: float) -> int | None:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Try JSON parse
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            return int(json.loads(m.group())["score"])
        except Exception:
            pass
    # Fallback: first digit 1-5
    m = re.search(r'[1-5]', text)
    return int(m.group()) if m else None


def validate_llm_vs_clan(df_sample: pd.DataFrame,
                          tokenizer, model) -> pd.DataFrame:
    rows = []
    for metric_key, mp in LLM_PROMPTS.items():
        clan_col   = mp["clan_col"]
        to_ordinal = mp["to_ordinal"]

        if clan_col not in df_sample.columns:
            console.print(f"  [yellow]Skipping LLM/{metric_key} — {clan_col} not found[/yellow]")
            continue

        clan_ordinals = []
        llm_scores    = []

        for _, row in tqdm(df_sample.iterrows(),
                           total=len(df_sample),
                           desc=f"LLM scoring: {metric_key}"):
            clan_val = row.get(clan_col)
            if pd.isna(clan_val):
                continue

            clan_ord = to_ordinal(float(clan_val))
            prompt   = build_prompt(str(row.get("clean_transcript", "")), metric_key)
            score    = llm_score(prompt, tokenizer, model,
                                  L["max_new_tokens"], L["temperature"])

            if score is not None:
                clan_ordinals.append(clan_ord)
                llm_scores.append(score)

        if len(clan_ordinals) < 5:
            continue

        clan_arr = np.array(clan_ordinals)
        llm_arr  = np.array(llm_scores)

        try:
            kappa = cohen_kappa_score(clan_arr.astype(int), llm_arr.astype(int),
                                       weights="quadratic")
        except Exception:
            kappa = np.nan

        r, p   = stats.pearsonr(clan_arr, llm_arr)
        mae    = float(np.mean(np.abs(clan_arr - llm_arr)))

        rows.append({
            "metric":        metric_key,
            "n_pairs":       len(clan_ordinals),
            "kappa":         round(float(kappa), 3),
            "pearson_r":     round(float(r),     3),
            "pearson_p":     round(float(p),     4),
            "mae":           round(mae,           3),
            "passes_kappa":  float(kappa) >= KAPPA_THRESH,
            "passes_r":      float(r)     >= PEARSON_THRESH,
            "_clan_arr":     clan_ordinals,
            "_llm_arr":      llm_scores,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def bland_altman(ax, a, b, title):
    mean  = (a + b) / 2
    diff  = a - b
    md, sd = np.mean(diff), np.std(diff)
    ax.scatter(mean, diff, alpha=0.4, s=25)
    ax.axhline(md,          color="red",  ls="--", lw=1.2, label=f"Bias={md:.2f}")
    ax.axhline(md + 1.96*sd, color="gray", ls=":",  lw=1,   label=f"±1.96SD")
    ax.axhline(md - 1.96*sd, color="gray", ls=":",  lw=1)
    ax.set_xlabel("Mean"); ax.set_ylabel("Difference"); ax.set_title(title)
    ax.legend(fontsize=7)


def scatter_with_r(ax, x, y, r, title):
    ax.scatter(x, y, alpha=0.4, s=25)
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(min(x), max(x), 50)
    ax.plot(xr, m*xr+b, "r--", lw=1.5)
    ax.set_title(f"{title}\nr={r:.3f}")
    ax.set_xlabel("CLAN value"); ax.set_ylabel("Python/LLM value")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 4: Three-Way Validation Against CLAN Ground Truth")

    master_path = PROC_DIR / "master_metrics.csv"
    clan_path   = PROC_DIR / "clan_metrics.csv"

    if not master_path.exists():
        console.print("[red]master_metrics.csv not found. Run 03_compute_metrics.py first.[/red]")
        return

    df_master = pd.read_csv(master_path)

    # ── Part A: Python vs CLAN ────────────────────────────────────────────
    console.rule("[cyan]Part A: Python NLP vs CLAN (ground truth)")

    if not clan_path.exists():
        console.print("[yellow]clan_metrics.csv not found — skipping Python vs CLAN comparison.[/yellow]")
        console.print("Run 02_clan_metrics.py if you have CLAN output files.")
        df_py_vs_clan = pd.DataFrame()
    else:
        df_clan = pd.read_csv(clan_path)
        # Rename CLAN columns to avoid clash with Python columns
        rename = {
            "mlu_words":       "mlu_words_clan",
            "ttr":             "ttr_clan",
            "ndw":             "ndw_clan",
            "vocd_d":          "vocd_d_clan",
            "maze_words_clan": "maze_words_clan",
        }
        df_clan = df_clan.rename(columns={k: v for k, v in rename.items() if k in df_clan.columns})
        df_merged = df_master.merge(df_clan[["participant_id"] + [v for v in rename.values() if v in df_clan.columns]],
                                     on="participant_id", how="inner")
        console.print(f"Merged {len(df_merged)} participants for Python vs CLAN comparison")

        df_py_vs_clan = validate_python_vs_clan(df_merged)
        df_py_vs_clan.to_csv(RES_DIR / "validation_python_vs_clan.csv", index=False)

        # Print table
        tbl = Table(title="Part A: Python vs CLAN Validation")
        tbl.add_column("Metric");   tbl.add_column("N"); tbl.add_column("Pearson r")
        tbl.add_column("MAE");      tbl.add_column("Bias"); tbl.add_column("Pass r≥0.7?")
        for _, r in df_py_vs_clan.iterrows():
            ok = "[green]✓[/green]" if r.get("passes_r") else "[red]✗[/red]"
            tbl.add_row(str(r["metric"]), str(r["n_pairs"]),
                        f"{r['pearson_r']:.3f}", f"{r['mae']:.3f}",
                        f"{r['mean_bias']:.3f}", ok)
        console.print(tbl)

        # Bland-Altman plots for Python vs CLAN
        if len(df_py_vs_clan) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for i, (_, row) in enumerate(df_py_vs_clan.iterrows()):
                if i >= 4: break
                py_col   = row["metric"]
                clan_col = PYTHON_TO_CLAN.get(py_col, "")
                if py_col in df_merged.columns and clan_col in df_merged.columns:
                    pair = df_merged[[py_col, clan_col]].dropna()
                    bland_altman(axes[i],
                                  pair[py_col].values,
                                  pair[clan_col].values,
                                  f"Python vs CLAN: {py_col}")
            plt.suptitle("Bland-Altman: Python NLP vs CLAN", fontsize=13)
            plt.tight_layout()
            plt.savefig(RES_DIR / "bland_altman_python_vs_clan.png", dpi=150)
            plt.close()

    # ── Part B: LLM vs CLAN ───────────────────────────────────────────────
    console.rule("[cyan]Part B: LLM vs CLAN (ground truth)")

    if not clan_path.exists():
        console.print("[yellow]Skipping LLM vs CLAN — no CLAN file.[/yellow]")
        df_llm_vs_clan = pd.DataFrame()
    else:
        try:
            tokenizer, model = load_llm(L["model_name"], L["device_map"])
        except Exception as e:
            console.print(f"[red]LLM load failed: {e}[/red]")
            console.print(f"Try changing model_name in config.yaml to 'google/flan-t5-large'")
            return

        sample_n   = min(SAMPLE_N, len(df_merged))
        df_sample  = df_merged.dropna(subset=["clean_transcript"]).sample(
            n=sample_n, random_state=CFG["project"]["seed"]
        ).reset_index(drop=True)
        console.print(f"Scoring {sample_n} transcripts with LLM...\n")

        df_llm_vs_clan = validate_llm_vs_clan(df_sample, tokenizer, model)

        # Drop internal arrays before saving
        save_cols = [c for c in df_llm_vs_clan.columns if not c.startswith("_")]
        df_llm_vs_clan[save_cols].to_csv(RES_DIR / "validation_llm_vs_clan.csv", index=False)

        tbl2 = Table(title="Part B: LLM vs CLAN Validation")
        tbl2.add_column("Metric");   tbl2.add_column("N");     tbl2.add_column("Kappa")
        tbl2.add_column("Pearson r"); tbl2.add_column("MAE"); tbl2.add_column("Pass κ≥0.6?")
        for _, r in df_llm_vs_clan.iterrows():
            ok = "[green]✓[/green]" if r.get("passes_kappa") else "[red]✗[/red]"
            tbl2.add_row(str(r["metric"]), str(r["n_pairs"]),
                          f"{r['kappa']:.3f}", f"{r['pearson_r']:.3f}",
                          f"{r['mae']:.3f}", ok)
        console.print(tbl2)

        # Scatter plots LLM vs CLAN
        if len(df_llm_vs_clan) > 0:
            n_plots = len(df_llm_vs_clan)
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            if n_plots == 1: axes = [axes]
            for i, (_, row) in enumerate(df_llm_vs_clan.iterrows()):
                if "_clan_arr" in row and "_llm_arr" in row:
                    scatter_with_r(axes[i],
                                    np.array(row["_clan_arr"]),
                                    np.array(row["_llm_arr"]),
                                    row["pearson_r"],
                                    row["metric"])
            plt.suptitle("LLM vs CLAN Score Scatter", fontsize=13)
            plt.tight_layout()
            plt.savefig(RES_DIR / "scatter_llm_vs_clan.png", dpi=150)
            plt.close()

    # ── Summary: which source to use for reward signal ────────────────────
    console.rule("[cyan]Summary: Reward Signal Recommendation")

    summary_rows = []
    for _, r in df_py_vs_clan.iterrows() if len(df_py_vs_clan) > 0 else []:
        summary_rows.append({
            "source": "Python NLP",
            "metric": r["metric"],
            "pearson_r": r["pearson_r"],
            "validated": r["passes_r"],
        })
    for _, r in df_llm_vs_clan.iterrows() if len(df_llm_vs_clan) > 0 else []:
        summary_rows.append({
            "source": "LLM",
            "metric": r["metric"],
            "pearson_r": r["pearson_r"],
            "validated": r.get("passes_kappa", False),
        })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(RES_DIR / "validation_summary.csv", index=False)

        py_pass  = (df_summary[df_summary.source == "Python NLP"]["validated"].mean()
                    if len(df_summary[df_summary.source == "Python NLP"]) > 0 else 0)
        llm_pass = (df_summary[df_summary.source == "LLM"]["validated"].mean()
                    if len(df_summary[df_summary.source == "LLM"]) > 0 else 0)

        console.print(f"\nPython NLP pass rate: [bold]{py_pass:.0%}[/bold] of metrics validated")
        console.print(f"LLM pass rate:        [bold]{llm_pass:.0%}[/bold] of metrics validated")

        if py_pass >= 0.75:
            console.print("\n[bold green]✓ Python NLP metrics are validated against CLAN ground truth.[/bold green]")
            console.print("[bold green]  Use Python-computed metrics as RL reward signal.[/bold green]")
        else:
            console.print("\n[bold yellow]⚠ Python metrics did not fully validate.[/bold yellow]")
            console.print("  Check individual metric results and consider adjusting your tokenizer.")

    console.print(f"\n[green]✓ Results saved to {RES_DIR}[/green]")


if __name__ == "__main__":
    main()