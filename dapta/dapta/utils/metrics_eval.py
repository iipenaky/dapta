"""
Statistical evaluation utilities for the DAPTA evaluation phase.

Implements:
  - Cohen's d effect size
  - Paired Wilcoxon signed-rank test with Bonferroni correction
  - Bootstrap confidence intervals
  - Inter-rater reliability (Cohen's κ, Pearson's r)

References
----------
Cohen, J. (1988). Statistical power analysis for the behavioural sciences (2nd ed.).
Virtanen et al. (2020). SciPy 1.0. Nature Methods.
Upton et al. (2024). iTalkBetter (d = 0.42 benchmark).
"""


import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats



# Effect size


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d between two independent groups.
    Positive = group_a > group_b (improvement).

    Benchmark from iTalkBetter (Upton et al., 2024): d = 0.42 is clinically meaningful.
    """
    n_a, n_b = len(group_a), len(group_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(group_a, ddof=1) + (n_b - 1) * np.var(group_b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def cohens_d_paired(before: np.ndarray, after: np.ndarray) -> float:
    """Cohen's d for paired (pre-post) data."""
    diff = after - before
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude (Cohen, 1988)."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"



# Hypothesis tests


def wilcoxon_test(
    before: np.ndarray,
    after: np.ndarray,
    alternative: str = "less",  # "less" = after > before (improvement)
) -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test for pre-post discourse metric change.

    Parameters
    ----------
    before      : Pre-therapy metric values
    after       : Post-therapy metric values
    alternative : "less" means H1: before < after (i.e., improvement)

    Returns
    -------
    (statistic, p_value)
    """
    stat, p = stats.wilcoxon(before, after, alternative=alternative)
    return float(stat), float(p)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction to a list of p-values.

    Returns
    -------
    (corrected_p_values, significant_flags)
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    significant = [p <= alpha for p in corrected]
    return corrected, significant


def run_pairwise_wilcoxon(
    metric_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """
    Run Wilcoxon tests across all discourse metrics with Bonferroni correction.

    Parameters
    ----------
    metric_results : {metric_name: (before_array, after_array)}
    alpha          : Significance threshold

    Returns
    -------
    Dict with test results per metric
    """
    metrics = list(metric_results.keys())
    raw_p_values = []
    stats_list = []
    ds = []

    for m in metrics:
        before, after = metric_results[m]
        stat, p = wilcoxon_test(before, after)
        d = cohens_d_paired(before, after)
        raw_p_values.append(p)
        stats_list.append(stat)
        ds.append(d)

    corrected_p, significant = bonferroni_correction(raw_p_values, alpha)

    results = {}
    for i, m in enumerate(metrics):
        results[m] = {
            "statistic": stats_list[i],
            "p_value_raw": raw_p_values[i],
            "p_value_corrected": corrected_p[i],
            "significant": significant[i],
            "cohens_d": ds[i],
            "effect_size": effect_size_label(ds[i]),
        }
    return results



# Bootstrap confidence intervals


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a scalar statistic.

    Parameters
    ----------
    data         : 1D array of observations
    statistic_fn : Function to compute the statistic (default: mean)
    n_bootstrap  : Number of bootstrap resamples
    ci           : Confidence level (default 0.95)
    random_seed  : For reproducibility

    Returns
    -------
    (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_seed)
    bootstrap_stats = np.array([
        statistic_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return lower, upper



# Inter-rater reliability


def cohens_kappa(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
) -> float:
    """
    Cohen's κ for categorical agreement between two raters (or auto vs manual).
    Used to validate DAE automated metrics against CLAN reference scores.
    """
    assert len(rater_a) == len(rater_b)
    categories = np.unique(np.concatenate([rater_a, rater_b]))
    n = len(rater_a)

    # Observed agreement
    p_o = np.mean(rater_a == rater_b)

    # Expected agreement
    p_e = sum(
        (np.sum(rater_a == c) / n) * (np.sum(rater_b == c) / n)
        for c in categories
    )

    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation coefficient and p-value.
    Used for continuous metric validation (DAE vs CLAN).
    """
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)



# Summary report


def format_evaluation_report(results: Dict[str, dict], alpha: float = 0.05) -> str:
    """
    Format a human-readable evaluation report from run_pairwise_wilcoxon output.
    """
    lines = [
        "=" * 70,
        "DAPTA DISCOURSE METRIC EVALUATION REPORT",
        f"Significance threshold (Bonferroni-corrected): α = {alpha}",
        f"Clinically meaningful effect size threshold: d ≥ 0.40 (Upton et al., 2024)",
        "=" * 70,
    ]
    for metric, res in results.items():
        sig = "✓ SIGNIFICANT" if res["significant"] else "✗ NOT SIGNIFICANT"
        meaningful = "✓ CLINICALLY MEANINGFUL" if abs(res["cohens_d"]) >= 0.4 else "○ BELOW THRESHOLD"
        lines += [
            f"\nMetric: {metric.upper()}",
            f"  Wilcoxon statistic : {res['statistic']:.4f}",
            f"  p-value (raw)      : {res['p_value_raw']:.4f}",
            f"  p-value (corrected): {res['p_value_corrected']:.4f}  {sig}",
            f"  Cohen's d          : {res['cohens_d']:+.3f}  ({res['effect_size']})  {meaningful}",
        ]
    lines.append("=" * 70)
    return "\n".join(lines)
