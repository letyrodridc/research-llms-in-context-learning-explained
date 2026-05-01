"""Build supplementary analyses for the paper:
  1) Classification accuracy and explanation-quality breakdowns by N and K.
  2) Spearman heatmap (judge metric vs accuracy) with Bonferroni-corrected
     significance markers.

Outputs are written to the same directory as this script.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent.parent
TRIALS_CSV = RUN_DIR / "trial_results.csv"
JUDGE_CSV = RUN_DIR / "judge_outputs" / "openai-gpt-5-mini" / "judge_results.csv"

DIMENSIONS = [
    "textual_groundedness",
    "hallucination_free",
    "concept_counting",
    "comprehensibility",
    "conciseness",
    "specificity",
    "discriminativeness",
    "instruction_following",
    "logical_coherence",
]

DIMENSION_LABELS = {
    "textual_groundedness": "Textual Groundedness (TG)",
    "hallucination_free": "Hallucination-Free (HF)",
    "concept_counting": "Concept Counting (CC)",
    "comprehensibility": "Comprehensibility (Co)",
    "conciseness": "Conciseness (Cn)",
    "specificity": "Specificity (Sp)",
    "discriminativeness": "Local Discriminativeness (LD)",
    "instruction_following": "Instruction Following (IF)",
    "logical_coherence": "Logical Coherence (LC)",
}

PROMPT_LABELS = {
    "nle": "NLE (E2)",
    "features": "Features (E3)",
    "rulebased": "Rule-Based (E4)",
    "axioms_ontology_v2": "DL Axioms (E5)",
}
EXPLANATION_PROMPTS_ORDER = ["nle", "features", "rulebased", "axioms_ontology_v2"]
ALL_PROMPTS_ORDER = ["classification", *EXPLANATION_PROMPTS_ORDER]
PROMPT_LABELS_ALL = {"classification": "Classification (E1)", **PROMPT_LABELS}


def _load_trials() -> pd.DataFrame:
    df = pd.read_csv(
        TRIALS_CSV,
        usecols=[
            "dataset", "prompt_type", "model",
            "config_n", "config_k", "config_q",
            "run_id", "query_index_within_episode", "correct",
        ],
    )
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)
    return df


def _load_judge() -> pd.DataFrame:
    cols = [
        "dataset", "prompt_type", "source_model",
        "config_n", "config_k", "config_q",
        "run_id", "query_index_within_episode",
        "judge_parse_error", *DIMENSIONS, "overall_score",
    ]
    df = pd.read_csv(JUDGE_CSV, usecols=cols)
    df = df[df["judge_parse_error"].isna()].copy()
    df = df.rename(columns={"source_model": "model"})
    for col in [*DIMENSIONS, "overall_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _wilcoxon_or_nan(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Paired Wilcoxon signed-rank test on x vs y. Returns (W, p, n_pairs)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    diffs = x - y
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 6:
        return float("nan"), float("nan"), int(len(nonzero))
    res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue), int(len(nonzero))


def _friedman_or_nan(per_n: dict[int, np.ndarray]) -> tuple[float, float, int]:
    levels = sorted(per_n.keys())
    arrays = [per_n[n] for n in levels]
    if len(arrays) < 3:
        return float("nan"), float("nan"), 0
    stacked = np.column_stack(arrays)
    mask = np.all(np.isfinite(stacked), axis=1)
    stacked = stacked[mask]
    if stacked.shape[0] < 6:
        return float("nan"), float("nan"), int(stacked.shape[0])
    res = friedmanchisquare(*[stacked[:, i] for i in range(stacked.shape[1])])
    return float(res.statistic), float(res.pvalue), int(stacked.shape[0])


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "p<10^{-4}"
    return f"p={p:.4f}"


def _stars(p: float, alpha_levels=(0.001, 0.01, 0.05)) -> str:
    if not np.isfinite(p):
        return ""
    a3, a2, a1 = alpha_levels
    if p < a3:
        return "***"
    if p < a2:
        return "**"
    if p < a1:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# (1) Classification accuracy by N and K
# ---------------------------------------------------------------------------
def analysis_accuracy_by_n_k(trials: pd.DataFrame) -> dict:
    out: dict = {}
    cell_keys = ["model", "dataset", "prompt_type", "config_n", "config_k"]
    cell = (
        trials.groupby(cell_keys)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_trials"})
        .reset_index()
    )
    out["cell"] = cell

    # marginal means + 95% CI across cells
    def _summary(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
        g = df.groupby(group)["accuracy"]
        m = g.mean()
        sd = g.std(ddof=1)
        n = g.count()
        se = sd / np.sqrt(n)
        return pd.DataFrame({
            "mean_accuracy": m,
            "sd": sd,
            "se": se,
            "ci95_low": m - 1.96 * se,
            "ci95_high": m + 1.96 * se,
            "n_cells": n,
        }).reset_index()

    out["by_n"] = _summary(cell, ["config_n"])
    out["by_k"] = _summary(cell, ["config_k"])
    out["by_n_k"] = _summary(cell, ["config_n", "config_k"])
    out["by_n_k_prompt"] = _summary(cell, ["config_n", "config_k", "prompt_type"])

    # Paired tests: K=1 vs K=5 at each N (paired across model x dataset x prompt)
    pivot_k = cell.pivot_table(
        index=["model", "dataset", "prompt_type", "config_n"],
        columns="config_k",
        values="accuracy",
    ).reset_index()
    rows = []
    for n in sorted(pivot_k["config_n"].unique()):
        sub = pivot_k[pivot_k["config_n"] == n]
        x = sub[5].to_numpy(dtype=float)
        y = sub[1].to_numpy(dtype=float)
        W, p, n_pairs = _wilcoxon_or_nan(x, y)
        rows.append({
            "config_n": int(n),
            "mean_acc_k1": float(np.nanmean(y)),
            "mean_acc_k5": float(np.nanmean(x)),
            "delta_k5_minus_k1": float(np.nanmean(x) - np.nanmean(y)),
            "wilcoxon_W": W, "p_value": p, "n_pairs": n_pairs,
        })
    # Pooled K=1 vs K=5 (across all N)
    x_all = pivot_k[5].to_numpy(dtype=float)
    y_all = pivot_k[1].to_numpy(dtype=float)
    W, p, n_pairs = _wilcoxon_or_nan(x_all, y_all)
    rows.append({
        "config_n": "pooled",
        "mean_acc_k1": float(np.nanmean(y_all)),
        "mean_acc_k5": float(np.nanmean(x_all)),
        "delta_k5_minus_k1": float(np.nanmean(x_all) - np.nanmean(y_all)),
        "wilcoxon_W": W, "p_value": p, "n_pairs": n_pairs,
    })
    out["k_test"] = pd.DataFrame(rows)

    # Friedman across N at each K (paired across model x dataset x prompt)
    pivot_n = cell.pivot_table(
        index=["model", "dataset", "prompt_type", "config_k"],
        columns="config_n",
        values="accuracy",
    ).reset_index()
    rows = []
    for k in sorted(pivot_n["config_k"].unique()):
        sub = pivot_n[pivot_n["config_k"] == k]
        per_n = {int(n): sub[n].to_numpy(dtype=float) for n in [2, 3, 4]}
        stat, p, n_pairs = _friedman_or_nan(per_n)
        rows.append({
            "config_k": int(k),
            "mean_acc_n2": float(np.nanmean(sub[2])),
            "mean_acc_n3": float(np.nanmean(sub[3])),
            "mean_acc_n4": float(np.nanmean(sub[4])),
            "friedman_chi2": stat, "p_value": p, "n_pairs": n_pairs,
        })
    # Pooled across K
    sub = pivot_n
    per_n = {int(n): sub[n].to_numpy(dtype=float) for n in [2, 3, 4]}
    stat, p, n_pairs = _friedman_or_nan(per_n)
    rows.append({
        "config_k": "pooled",
        "mean_acc_n2": float(np.nanmean(sub[2])),
        "mean_acc_n3": float(np.nanmean(sub[3])),
        "mean_acc_n4": float(np.nanmean(sub[4])),
        "friedman_chi2": stat, "p_value": p, "n_pairs": n_pairs,
    })
    out["n_test"] = pd.DataFrame(rows)

    # Pairwise N comparisons (Wilcoxon, Bonferroni corrected: 3 comparisons)
    rows = []
    for k in [1, 5, "pooled"]:
        if k == "pooled":
            sub = pivot_n
        else:
            sub = pivot_n[pivot_n["config_k"] == k]
        for a, b in combinations([2, 3, 4], 2):
            xa = sub[a].to_numpy(dtype=float)
            xb = sub[b].to_numpy(dtype=float)
            W, p, n_pairs = _wilcoxon_or_nan(xa, xb)
            rows.append({
                "config_k": k,
                "n_a": a, "n_b": b,
                "mean_a": float(np.nanmean(xa)),
                "mean_b": float(np.nanmean(xb)),
                "delta_a_minus_b": float(np.nanmean(xa) - np.nanmean(xb)),
                "wilcoxon_W": W, "p_raw": p,
                "p_bonferroni": min(1.0, p * 3) if np.isfinite(p) else float("nan"),
                "n_pairs": n_pairs,
            })
    out["n_pairwise"] = pd.DataFrame(rows)

    return out


def plot_accuracy_by_n_k(out: dict, save_path: Path) -> None:
    by_n_k = out["by_n_k"].copy()
    by_n_k_prompt = out["by_n_k_prompt"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: pooled
    ax = axes[0]
    n_vals = sorted(by_n_k["config_n"].unique())
    for k, color, marker in [(1, "#1f77b4", "o"), (5, "#d62728", "s")]:
        sub = by_n_k[by_n_k["config_k"] == k].sort_values("config_n")
        ax.errorbar(
            sub["config_n"], sub["mean_accuracy"],
            yerr=1.96 * sub["se"], fmt=f"-{marker}",
            color=color, label=f"K={k}", capsize=3, lw=1.6, ms=7,
        )
    ax.set_xlabel("N (number of classes)")
    ax.set_ylabel("Mean classification accuracy")
    ax.set_xticks(n_vals)
    ax.set_ylim(0.78, 1.02)
    ax.set_title("(a) Accuracy vs N, by K\n(pooled across models, datasets, prompts)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Support shots", loc="lower left")

    # Right: lines per prompt, K=1 dashed and K=5 solid
    ax = axes[1]
    prompts = ALL_PROMPTS_ORDER
    colors = {p: plt.cm.tab10(i) for i, p in enumerate(prompts)}
    for prompt in prompts:
        for k, ls, marker in [(1, "--", "o"), (5, "-", "s")]:
            sub = by_n_k_prompt[
                (by_n_k_prompt["prompt_type"] == prompt)
                & (by_n_k_prompt["config_k"] == k)
            ].sort_values("config_n")
            ax.errorbar(
                sub["config_n"], sub["mean_accuracy"],
                yerr=1.96 * sub["se"], fmt=ls + marker,
                color=colors[prompt], capsize=2, lw=1.3, ms=5,
                alpha=0.9,
                label=f"{PROMPT_LABELS_ALL[prompt]} K={k}",
            )
    ax.set_xticks(n_vals)
    ax.set_xlabel("N (number of classes)")
    ax.set_ylabel("Mean classification accuracy")
    ax.set_ylim(0.6, 1.03)
    ax.set_title("(b) Accuracy by N, K and prompt type\n(dashed = K=1, solid = K=5)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (2) Explanation metrics by N and K
# ---------------------------------------------------------------------------
def analysis_explanation_by_n_k(judge: pd.DataFrame) -> dict:
    out: dict = {}
    cell_keys = ["model", "dataset", "prompt_type", "config_n", "config_k"]
    metrics = ["overall_score", *DIMENSIONS]
    cell = judge.groupby(cell_keys)[metrics].mean().reset_index()
    out["cell"] = cell

    def _summary(df: pd.DataFrame, group: list[str], metric: str) -> pd.DataFrame:
        g = df.groupby(group)[metric]
        m, sd, n = g.mean(), g.std(ddof=1), g.count()
        se = sd / np.sqrt(n)
        res = pd.DataFrame({
            "metric": metric,
            "mean": m, "sd": sd, "se": se,
            "ci95_low": m - 1.96 * se, "ci95_high": m + 1.96 * se,
            "n_cells": n,
        }).reset_index()
        return res

    out["by_n_overall"] = _summary(cell, ["config_n"], "overall_score")
    out["by_k_overall"] = _summary(cell, ["config_k"], "overall_score")
    out["by_n_k_overall"] = _summary(cell, ["config_n", "config_k"], "overall_score")
    out["by_n_k_prompt_overall"] = _summary(
        cell, ["config_n", "config_k", "prompt_type"], "overall_score",
    )

    # Per-dimension marginals
    rows = []
    for m in metrics:
        for n in [2, 3, 4]:
            for k in [1, 5]:
                sub = cell[(cell["config_n"] == n) & (cell["config_k"] == k)]
                vals = sub[m].dropna().to_numpy()
                rows.append({
                    "metric": m, "config_n": n, "config_k": k,
                    "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                    "n_cells": int(len(vals)),
                })
    out["by_n_k_metric"] = pd.DataFrame(rows)

    # Paired tests: K=1 vs K=5 at each N for overall_score and per dimension
    pivot_k = cell.pivot_table(
        index=["model", "dataset", "prompt_type", "config_n"],
        columns="config_k",
        values=metrics,
    )
    rows = []
    for metric in metrics:
        for n in [2, 3, 4, "pooled"]:
            if n == "pooled":
                sub = pivot_k[metric]
            else:
                sub = pivot_k[metric].xs(n, level="config_n")
            x = sub[5].to_numpy(dtype=float)
            y = sub[1].to_numpy(dtype=float)
            W, p, n_pairs = _wilcoxon_or_nan(x, y)
            rows.append({
                "metric": metric, "config_n": n,
                "mean_k1": float(np.nanmean(y)), "mean_k5": float(np.nanmean(x)),
                "delta_k5_minus_k1": float(np.nanmean(x) - np.nanmean(y)),
                "wilcoxon_W": W, "p_raw": p,
                "n_pairs": n_pairs,
            })
    df_k = pd.DataFrame(rows)
    # Bonferroni correction: 10 metrics × 4 N-levels = 40 comparisons
    n_tests = len(df_k)
    df_k["p_bonferroni"] = (df_k["p_raw"] * n_tests).clip(upper=1.0)
    out["k_test"] = df_k

    # Friedman across N at each K
    pivot_n = cell.pivot_table(
        index=["model", "dataset", "prompt_type", "config_k"],
        columns="config_n",
        values=metrics,
    )
    rows = []
    for metric in metrics:
        for k in [1, 5, "pooled"]:
            if k == "pooled":
                sub = pivot_n[metric]
            else:
                sub = pivot_n[metric].xs(k, level="config_k")
            per_n = {int(nn): sub[nn].to_numpy(dtype=float) for nn in [2, 3, 4]}
            stat, p, n_pairs = _friedman_or_nan(per_n)
            rows.append({
                "metric": metric, "config_k": k,
                "mean_n2": float(np.nanmean(sub[2])),
                "mean_n3": float(np.nanmean(sub[3])),
                "mean_n4": float(np.nanmean(sub[4])),
                "friedman_chi2": stat, "p_raw": p,
                "n_pairs": n_pairs,
            })
    df_n = pd.DataFrame(rows)
    n_tests = len(df_n)
    df_n["p_bonferroni"] = (df_n["p_raw"] * n_tests).clip(upper=1.0)
    out["n_test"] = df_n

    return out


def plot_explanation_by_n_k(out: dict, save_path: Path) -> None:
    by_n_k_prompt = out["by_n_k_prompt_overall"].copy()
    by_n_k_metric = out["by_n_k_metric"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: overall judge score as line plot per prompt × K
    ax = axes[0]
    n_vals = [2, 3, 4]
    prompts = EXPLANATION_PROMPTS_ORDER
    colors = {p: plt.cm.tab10(i) for i, p in enumerate(prompts)}
    for prompt in prompts:
        for k, ls, marker in [(1, "--", "o"), (5, "-", "s")]:
            sub = by_n_k_prompt[
                (by_n_k_prompt["prompt_type"] == prompt)
                & (by_n_k_prompt["config_k"] == k)
            ].sort_values("config_n")
            ax.errorbar(
                sub["config_n"], sub["mean"],
                yerr=1.96 * sub["se"], fmt=ls + marker,
                color=colors[prompt], capsize=2, lw=1.4, ms=6, alpha=0.95,
                label=f"{PROMPT_LABELS[prompt]} K={k}",
            )
    ax.set_xticks(n_vals)
    ax.set_xlabel("N (number of classes)")
    ax.set_ylabel("Mean overall judge score (1–5)")
    ax.set_ylim(3.0, 5.0)
    ax.set_title("(a) Overall explanation score by N, K and prompt\n"
                 "(dashed = K=1, solid = K=5)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)

    # Right: per-dimension means by K (pooled across N, prompts)
    ax = axes[1]
    dims = DIMENSIONS
    means_k1 = [
        by_n_k_metric[
            (by_n_k_metric["metric"] == d) & (by_n_k_metric["config_k"] == 1)
        ]["mean"].mean() for d in dims
    ]
    means_k5 = [
        by_n_k_metric[
            (by_n_k_metric["metric"] == d) & (by_n_k_metric["config_k"] == 5)
        ]["mean"].mean() for d in dims
    ]
    pos = np.arange(len(dims))
    ax.barh(pos - 0.2, means_k1, height=0.4, color="#1f77b4", alpha=0.85, label="K=1")
    ax.barh(pos + 0.2, means_k5, height=0.4, color="#d62728", alpha=0.85, label="K=5")
    ax.set_yticks(pos)
    ax.set_yticklabels([DIMENSION_LABELS[d] for d in dims], fontsize=8)
    ax.set_xlim(2.5, 5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean score (1–5)")
    ax.set_title("(b) Explanation dimensions by K\n(pooled across N and prompts)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    # annotate deltas
    for i, (m1, m5) in enumerate(zip(means_k1, means_k5)):
        delta = m5 - m1
        ax.text(
            max(m1, m5) + 0.03, i,
            f"Δ={delta:+.2f}",
            va="center", fontsize=7.5, color="black",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quad_panel(
    acc_out: dict,
    expl_out: dict,
    judge: pd.DataFrame,
    trials: pd.DataFrame,
    save_path: Path,
) -> None:
    """4-panel summary figure (1 row × 4 cols):
      A) Accuracy vs N, two curves (one per K).
      B) Mean explanation-dim score vs K (one line per dimension), LD highlighted.
      C) Mean explanation-dim score vs N (one line per dimension), LD highlighted.
      D) Mean classification accuracy vs LD (binned by integer LD score).
    Panels B and C share the same y-axis (set to identical limits).
    """
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.0))
    axA, axB, axC, axD = axes

    # ----- Panel A: accuracy vs N, by K -----
    # Colors chosen to avoid clashing with LD's red in panels B–D.
    by_n_k = acc_out["by_n_k"].copy()
    n_vals = sorted(by_n_k["config_n"].unique())
    for k, color, marker in [(1, "#4575b4", "o"), (5, "#1a9850", "s")]:
        sub = by_n_k[by_n_k["config_k"] == k].sort_values("config_n")
        axA.errorbar(
            sub["config_n"], sub["mean_accuracy"],
            yerr=1.96 * sub["se"], fmt=f"-{marker}",
            color=color, label=f"K={k}", capsize=3, lw=2.0, ms=8,
        )
    axA.set_xlabel("N (number of classes)")
    axA.set_ylabel("Mean classification accuracy")
    axA.set_xticks(n_vals)
    axA.set_ylim(0.78, 1.02)
    axA.set_title("(A) Accuracy vs N, by K")
    axA.grid(True, alpha=0.3)
    axA.legend(title="Support shots", loc="lower left")

    # ----- Shared dimension styling for panels B and C -----
    cell = expl_out["cell"].copy()
    LD = "discriminativeness"
    other_dims = [d for d in DIMENSIONS if d != LD]
    cmap = plt.cm.tab10
    other_colors = {d: cmap(i % 10) for i, d in enumerate(other_dims)}
    LD_COLOR = "#e6194B"  # bright red

    def _plot_dims_vs_axis(ax, x_field, x_values, x_label, title):
        # Other dimensions first (muted)
        for d in other_dims:
            grouped = cell.groupby(x_field)[d].agg(["mean", "std", "count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            grouped = grouped.reindex(x_values)
            ax.errorbar(
                x_values, grouped["mean"],
                yerr=1.96 * grouped["se"],
                fmt="-o", color=other_colors[d], capsize=2,
                lw=1.0, ms=4, alpha=0.55,
                label=DIMENSION_LABELS[d],
            )
        # LD on top, bold
        grouped = cell.groupby(x_field)[LD].agg(["mean", "std", "count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
        grouped = grouped.reindex(x_values)
        ax.errorbar(
            x_values, grouped["mean"],
            yerr=1.96 * grouped["se"],
            fmt="-D", color=LD_COLOR, capsize=3,
            lw=3.0, ms=9, alpha=1.0,
            label=DIMENSION_LABELS[LD],
            zorder=10,
        )
        ax.set_xticks(x_values)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # ----- Panel B: dims vs K (pooled across N) -----
    _plot_dims_vs_axis(
        axB, "config_k", [1, 5],
        "K (support shots per class)",
        "(B) Explanation dimensions vs K",
    )
    axB.set_ylabel("Mean score (1–5)")

    # ----- Panel C: dims vs N (pooled across K) -----
    _plot_dims_vs_axis(
        axC, "config_n", [2, 3, 4],
        "N (number of classes)",
        "(C) Explanation dimensions vs N",
    )

    # Shared y-axis range for B and C
    y_min = min(axB.get_ylim()[0], axC.get_ylim()[0])
    y_max = max(axB.get_ylim()[1], axC.get_ylim()[1])
    axB.set_ylim(y_min, y_max)
    axC.set_ylim(y_min, y_max)
    axC.tick_params(labelleft=False)

    # Single legend for B and C (placed below them)
    handles, labels = axB.get_legend_handles_labels()
    # Move LD to the front of the legend
    ld_idx = next(i for i, lab in enumerate(labels) if "Discriminativeness" in lab)
    handles = [handles[ld_idx]] + [h for i, h in enumerate(handles) if i != ld_idx]
    labels = [labels[ld_idx]] + [l for i, l in enumerate(labels) if i != ld_idx]
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5, fontsize=8,
        bbox_to_anchor=(0.55, -0.06), frameon=False,
    )

    # ----- Panel D: accuracy vs LD bin -----
    join_keys = [
        "model", "dataset", "prompt_type",
        "config_n", "config_k", "config_q",
        "run_id", "query_index_within_episode",
    ]
    merged = judge.merge(trials[[*join_keys, "correct"]], on=join_keys, how="inner")
    merged = merged.dropna(subset=[LD, "correct"])
    merged["LD_bin"] = merged[LD].round().astype(int)

    rows = []
    for ld in sorted(merged["LD_bin"].unique()):
        sub = merged[merged["LD_bin"] == ld]
        n = len(sub)
        p = float(sub["correct"].mean())
        # Wilson 95% CI for proportion
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        rows.append({
            "LD": int(ld), "n_trials": n,
            "mean_accuracy": p,
            "ci95_low": center - half, "ci95_high": center + half,
        })
    acc_vs_ld = pd.DataFrame(rows)

    axD.errorbar(
        acc_vs_ld["LD"], acc_vs_ld["mean_accuracy"],
        yerr=[
            acc_vs_ld["mean_accuracy"] - acc_vs_ld["ci95_low"],
            acc_vs_ld["ci95_high"] - acc_vs_ld["mean_accuracy"],
        ],
        fmt="-D", color=LD_COLOR, capsize=4, lw=2.5, ms=9,
    )
    for _, r in acc_vs_ld.iterrows():
        axD.text(
            r["LD"], min(0.99, r["ci95_high"] + 0.01),
            f"n={int(r['n_trials'])}",
            ha="center", va="bottom", fontsize=8,
        )
    axD.set_xticks([1, 2, 3, 4, 5])
    axD.set_xlim(0.7, 5.3)
    axD.set_ylim(0.5, 1.02)
    axD.set_xlabel("Local Discriminativeness score")
    axD.set_ylabel("Mean classification accuracy")
    axD.set_title("(D) Accuracy vs LD")
    axD.grid(True, alpha=0.3)

    # Save the per-bin table for the supplement
    acc_vs_ld.to_csv(save_path.with_name("accuracy_vs_LD_bin.csv"), index=False)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_explanation_dim_vs_n_single(out: dict, save_path: Path) -> None:
    """Single panel: one line per dimension, x = N, K-pooled.
    Significant N-trends (Bonferroni) are bolded in the legend with a star."""
    cell = out["cell"].copy()
    n_test = out["n_test"]

    dims = DIMENSIONS
    n_vals = [2, 3, 4]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = plt.cm.tab10
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h"]

    for i, dim in enumerate(dims):
        grouped = cell.groupby("config_n")[dim].agg(["mean", "std", "count"])
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
        grouped = grouped.reindex(n_vals)

        row = n_test[(n_test["metric"] == dim) & (n_test["config_k"] == "pooled")]
        p_b = float(row["p_bonferroni"].iloc[0]) if not row.empty else float("nan")
        stars = _stars(p_b) if np.isfinite(p_b) else ""
        sig = bool(stars)

        label = f"{DIMENSION_LABELS[dim]} {stars}".strip()
        ax.errorbar(
            n_vals, grouped["mean"],
            yerr=1.96 * grouped["se"],
            fmt="-" + markers[i],
            color=cmap(i), capsize=2,
            lw=2.0 if sig else 1.0,
            ms=7 if sig else 5,
            alpha=1.0 if sig else 0.55,
            label=label,
        )

    ax.set_xticks(n_vals)
    ax.set_xlabel("N (number of classes)")
    ax.set_ylabel("Mean score (1–5), pooled across K, models, datasets, prompts")
    ax.set_title(
        "Effect of N on each explanation dimension\n"
        "(thick lines = Bonferroni-significant N-trend; "
        "* p<0.05, ** p<0.01, *** p<0.001 over 30 N-tests)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=9, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_explanation_dim_vs_n(out: dict, save_path: Path) -> None:
    """Small multiples: one panel per dimension, mean ± 95 % CI across N,
    with K=1 dashed and K=5 solid. Annotates each panel with the
    Bonferroni-corrected p-value of the pooled Friedman N-test."""
    cell = out["cell"].copy()
    n_test = out["n_test"]

    dims = DIMENSIONS
    n_vals = [2, 3, 4]

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5), sharex=True)
    axes_flat = axes.flatten()

    # Per-dim K-pooled (model, dataset, prompt, K) cells aggregated across N
    for idx, dim in enumerate(dims):
        ax = axes_flat[idx]
        for k, ls, marker, color in [(1, "--", "o", "#1f77b4"),
                                     (5, "-",  "s", "#d62728")]:
            sub = cell[cell["config_k"] == k]
            grouped = sub.groupby("config_n")[dim].agg(["mean", "std", "count"])
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
            grouped = grouped.reindex(n_vals)
            ax.errorbar(
                n_vals, grouped["mean"],
                yerr=1.96 * grouped["se"], fmt=ls + marker,
                color=color, capsize=2, lw=1.4, ms=5,
                label=f"K={k}",
            )
        # title with significance
        row = n_test[(n_test["metric"] == dim) & (n_test["config_k"] == "pooled")]
        if not row.empty:
            p_b = float(row["p_bonferroni"].iloc[0])
            stars = _stars(p_b)
            if np.isfinite(p_b) and p_b < 1e-4:
                pstr = "p<10⁻⁴"
            else:
                pstr = f"p={p_b:.3f}" if np.isfinite(p_b) else ""
            title = f"{DIMENSION_LABELS[dim]}\nFriedman N: {pstr} {stars}"
        else:
            title = DIMENSION_LABELS[dim]
        ax.set_title(title, fontsize=9)
        ax.set_xticks(n_vals)
        ax.grid(True, alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Mean score (1–5)")
        if idx >= 6:
            ax.set_xlabel("N (number of classes)")
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "Effect of N on each explanation dimension\n"
        "(small markers = mean ± 95 % CI across model × dataset × prompt cells; "
        "p-values Bonferroni-corrected over the 30 N-tests of the supplementary CSV)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (3) Spearman heatmap with significance
# ---------------------------------------------------------------------------
def analysis_spearman_significance(judge: pd.DataFrame, trials: pd.DataFrame) -> dict:
    join_keys = [
        "model", "dataset", "prompt_type",
        "config_n", "config_k", "config_q",
        "run_id", "query_index_within_episode",
    ]
    merged = judge.merge(trials[[*join_keys, "correct"]], on=join_keys, how="inner")

    rows = []
    for prompt in EXPLANATION_PROMPTS_ORDER:
        sub = merged[merged["prompt_type"] == prompt]
        for dim in DIMENSIONS:
            x = sub[dim].to_numpy(dtype=float)
            y = sub["correct"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 3 or np.std(y) == 0 or np.std(x) == 0:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(x, y)
            rows.append({
                "prompt_type": prompt, "dimension": dim,
                "n_matched": int(len(x)),
                "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
                "p_raw": float(p) if np.isfinite(p) else float("nan"),
            })
    df = pd.DataFrame(rows)
    n_tests = df["p_raw"].notna().sum()
    df["p_bonferroni"] = (df["p_raw"] * n_tests).clip(upper=1.0)
    df["sig_05"] = df["p_bonferroni"] < 0.05
    df["sig_01"] = df["p_bonferroni"] < 0.01
    df["sig_001"] = df["p_bonferroni"] < 0.001
    df["stars"] = df["p_bonferroni"].apply(_stars)
    return {"table": df, "n_tests": int(n_tests)}


def plot_spearman_heatmap(res: dict, save_path: Path) -> None:
    df = res["table"]
    rows = DIMENSIONS
    cols = EXPLANATION_PROMPTS_ORDER
    rho_mat = np.full((len(rows), len(cols)), np.nan)
    star_mat = np.empty((len(rows), len(cols)), dtype=object)
    star_mat[:] = ""
    for r, dim in enumerate(rows):
        for c, prompt in enumerate(cols):
            cell = df[(df["dimension"] == dim) & (df["prompt_type"] == prompt)]
            if not cell.empty:
                rho_mat[r, c] = cell["spearman_rho"].iloc[0]
                star_mat[r, c] = cell["stars"].iloc[0]

    cmap = LinearSegmentedColormap.from_list(
        "rho", ["#2166ac", "#f7f7f7", "#b2182b"], N=256,
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    vmax = max(0.4, np.nanmax(np.abs(rho_mat)))
    im = ax.imshow(rho_mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([PROMPT_LABELS[p] for p in cols], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([DIMENSION_LABELS[d] for d in rows])
    for r in range(len(rows)):
        for c in range(len(cols)):
            val = rho_mat[r, c]
            if np.isnan(val):
                continue
            stars = star_mat[r, c]
            txt = f"{val:.2f}{stars}"
            color = "white" if abs(val) > 0.25 else "black"
            ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=9)
    ax.set_title(
        "Spearman ρ between judge metrics and classification accuracy\n"
        "* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni-corrected, 36 tests)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman ρ")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    trials = _load_trials()
    judge = _load_judge()
    print(f"  trials: {len(trials):,} rows")
    print(f"  judge:  {len(judge):,} rows (after dropping parse errors)")

    print("\n[1/3] Classification accuracy by N and K...")
    acc = analysis_accuracy_by_n_k(trials)
    acc["cell"].to_csv(HERE / "accuracy_cells_by_model_dataset_prompt_n_k.csv", index=False)
    acc["by_n"].to_csv(HERE / "accuracy_marginals_by_n.csv", index=False)
    acc["by_k"].to_csv(HERE / "accuracy_marginals_by_k.csv", index=False)
    acc["by_n_k"].to_csv(HERE / "accuracy_marginals_by_n_k.csv", index=False)
    acc["by_n_k_prompt"].to_csv(HERE / "accuracy_by_n_k_prompt.csv", index=False)
    acc["k_test"].to_csv(HERE / "accuracy_k_paired_tests.csv", index=False)
    acc["n_test"].to_csv(HERE / "accuracy_n_friedman_tests.csv", index=False)
    acc["n_pairwise"].to_csv(HERE / "accuracy_n_pairwise_tests.csv", index=False)
    plot_accuracy_by_n_k(acc, HERE / "fig_accuracy_by_n_k.png")
    print("  -> fig_accuracy_by_n_k.png")

    print("\n[2/3] Explanation metrics by N and K...")
    expl = analysis_explanation_by_n_k(judge)
    expl["cell"].to_csv(
        HERE / "explanation_cells_by_model_dataset_prompt_n_k.csv", index=False,
    )
    expl["by_n_k_overall"].to_csv(HERE / "explanation_overall_by_n_k.csv", index=False)
    expl["by_n_k_prompt_overall"].to_csv(
        HERE / "explanation_overall_by_n_k_prompt.csv", index=False,
    )
    expl["by_n_k_metric"].to_csv(HERE / "explanation_by_n_k_metric.csv", index=False)
    expl["k_test"].to_csv(HERE / "explanation_k_paired_tests.csv", index=False)
    expl["n_test"].to_csv(HERE / "explanation_n_friedman_tests.csv", index=False)
    plot_explanation_by_n_k(expl, HERE / "fig_explanation_by_n_k.png")
    print("  -> fig_explanation_by_n_k.png")
    plot_explanation_dim_vs_n(expl, HERE / "fig_explanation_dim_vs_n.png")
    print("  -> fig_explanation_dim_vs_n.png")
    plot_explanation_dim_vs_n_single(expl, HERE / "fig_explanation_dim_vs_n_single.png")
    print("  -> fig_explanation_dim_vs_n_single.png")
    plot_quad_panel(acc, expl, judge, trials, HERE / "fig_quad_panel.png")
    print("  -> fig_quad_panel.png")

    print("\n[3/3] Spearman heatmap with significance markers...")
    spear = analysis_spearman_significance(judge, trials)
    spear["table"].to_csv(HERE / "spearman_with_significance.csv", index=False)
    plot_spearman_heatmap(spear, HERE / "fig_spearman_heatmap_signif.png")
    print("  -> fig_spearman_heatmap_signif.png")
    print(f"  Bonferroni n_tests = {spear['n_tests']}")

    summary = {
        "accuracy": {
            "by_n_k": acc["by_n_k"].round(4).to_dict(orient="records"),
            "k_test": acc["k_test"].round(6).to_dict(orient="records"),
            "n_test": acc["n_test"].round(6).to_dict(orient="records"),
            "n_pairwise": acc["n_pairwise"].round(6).to_dict(orient="records"),
        },
        "explanation": {
            "k_test": expl["k_test"].round(6).to_dict(orient="records"),
            "n_test": expl["n_test"].round(6).to_dict(orient="records"),
        },
        "spearman_significance": {
            "n_tests_bonferroni": spear["n_tests"],
            "table": spear["table"].round(6).to_dict(orient="records"),
        },
    }
    with (HERE / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote summary.json")


if __name__ == "__main__":
    main()
