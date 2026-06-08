"""
Generate the missing plots for the judge validation appendix.
Saves to analysis/plots/ alongside existing figures.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

GPT_CSV = BASE.parent / "openai-gpt-5-mini" / "judge_results.csv"
QO_CSV  = BASE.parent / "qwen3-vl-32b-thinking-local-query_only"  / "judge_results.csv"
QAS_CSV = BASE.parent / "qwen3-vl-32b-thinking-local-query_and_support" / "judge_results.csv"
SPEARMAN_CSV = BASE / "spearman_results.csv"

# ── constants ──────────────────────────────────────────────────────────────
METRICS = [
    "textual_groundedness", "hallucination_free", "concept_counting",
    "comprehensibility", "conciseness", "specificity",
    "discriminativeness", "instruction_following", "logical_coherence",
]
METRIC_LABELS = {
    "textual_groundedness":  "Textual\nGroundedness",
    "hallucination_free":    "Hallucination\nFree",
    "concept_counting":      "Concept\nCounting",
    "comprehensibility":     "Comprehen-\nsibility",
    "conciseness":           "Conciseness",
    "specificity":           "Specificity",
    "discriminativeness":    "Local\nDiscriminat.",
    "instruction_following": "Instruction\nFollowing",
    "logical_coherence":     "Logical\nCoherence",
}
METRIC_LABELS_SHORT = {
    "textual_groundedness":  "Textual Groundedness",
    "hallucination_free":    "Hallucination Free",
    "concept_counting":      "Concept Counting",
    "comprehensibility":     "Comprehensibility",
    "conciseness":           "Conciseness",
    "specificity":           "Specificity",
    "discriminativeness":    "Local Discriminativeness",
    "instruction_following": "Instruction Following",
    "logical_coherence":     "Logical Coherence",
}
KEY_COLS = [
    "source_model", "dataset", "prompt_type",
    "config_n", "config_k", "run_id", "query_index_within_episode",
]

# Group membership for coloring
GROUP_ROBUST      = {"comprehensibility", "specificity", "discriminativeness"}
GROUP_SENSITIVE   = {"textual_groundedness", "hallucination_free"}
GROUP_INTERMEDIATE = {"conciseness", "instruction_following", "logical_coherence"}
GROUP_OTHER       = {"concept_counting"}

GROUP_COLOR = {
    "robust":       "#2196F3",   # blue
    "sensitive":    "#E53935",   # red
    "intermediate": "#FB8C00",   # orange
    "other":        "#757575",   # grey
}
GROUP_LABEL = {
    "robust":       "Robust to visual access",
    "sensitive":    "Sensitive to visual access",
    "intermediate": "Intermediate (form-dependent)",
    "other":        "Unreliable cross-model",
}

def metric_group(m):
    if m in GROUP_ROBUST:       return "robust"
    if m in GROUP_SENSITIVE:    return "sensitive"
    if m in GROUP_INTERMEDIATE: return "intermediate"
    return "other"

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}

# ── load & merge data ──────────────────────────────────────────────────────
def load_merged():
    gpt = pd.read_csv(GPT_CSV)
    qo  = pd.read_csv(QO_CSV)
    qas = pd.read_csv(QAS_CSV)
    for df in [gpt, qo, qas]:
        for m in METRICS:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    gpt = gpt.dropna(subset=METRICS)
    qo  = qo.dropna(subset=METRICS)
    qas = qas.dropna(subset=METRICS)
    m = gpt.merge(qo[KEY_COLS + METRICS], on=KEY_COLS, suffixes=("_gpt", "_qo"))
    m = m.merge(qas[KEY_COLS + METRICS], on=KEY_COLS, suffixes=("", "_qas"))
    # rename plain metric cols → _qas suffix
    rename = {col: col + "_qas" for col in METRICS if col in m.columns}
    m = m.rename(columns=rename)
    return m

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Spearman ρ grouped bar chart (all 9 metrics × 3 pairs)
# ══════════════════════════════════════════════════════════════════════════
def plot_rho_barplot(spearman_df):
    with plt.rc_context(STYLE):
        sub = spearman_df[spearman_df["subset"] == "all"].copy()
        pair_labels = {
            "A-B": "GPT-QO vs Qwen-QO\n(cross-model, no support images)",
            "A-C": "GPT-QO vs Qwen-QAS\n(cross-model, different visual access)",
            "B-C": "Qwen-QO vs Qwen-QAS\n(same model, ± support images)",
        }
        pairs = ["B-C", "A-B", "A-C"]
        pair_colors = {"B-C": "#1565C0", "A-B": "#546E7A", "A-C": "#78909C"}

        # Order metrics by group, then by B-C ρ within group
        order_groups = ["robust", "sensitive", "other", "intermediate"]
        bc = sub[sub["pair"] == "B-C"].set_index("metric")["spearman_r"]
        metric_order = []
        for g in order_groups:
            ms = [m for m in METRICS if metric_group(m) == g]
            ms.sort(key=lambda m: -bc.get(m, 0))
            metric_order.extend(ms)

        n_metrics = len(metric_order)
        n_pairs   = len(pairs)
        bar_w = 0.22
        x = np.arange(n_metrics)

        fig, ax = plt.subplots(figsize=(13, 5))

        offsets = [-bar_w, 0, bar_w]
        for i, pair in enumerate(pairs):
            pdf = sub[sub["pair"] == pair].set_index("metric")
            rhos   = [pdf.loc[m, "spearman_r"]  if m in pdf.index else np.nan for m in metric_order]
            ci_lo  = [pdf.loc[m, "ci_lower"]     if m in pdf.index else np.nan for m in metric_order]
            ci_hi  = [pdf.loc[m, "ci_upper"]     if m in pdf.index else np.nan for m in metric_order]
            yerr_lo = [r - lo for r, lo in zip(rhos, ci_lo)]
            yerr_hi = [hi - r  for r, hi in zip(rhos, ci_hi)]
            bars = ax.bar(x + offsets[i], rhos, bar_w,
                          label=pair_labels[pair],
                          color=pair_colors[pair],
                          alpha=0.85,
                          error_kw=dict(elinewidth=1, capsize=3))
            ax.errorbar(x + offsets[i], rhos,
                        yerr=[yerr_lo, yerr_hi],
                        fmt="none", color="black", elinewidth=1, capsize=3)

        # Group separators and labels
        group_spans = {}
        for m in metric_order:
            g = metric_group(m)
            idx = metric_order.index(m)
            if g not in group_spans:
                group_spans[g] = [idx, idx]
            else:
                group_spans[g][1] = idx

        for g, (start, end) in group_spans.items():
            mid = (start + end) / 2
            ax.text(mid, 1.01, GROUP_LABEL[g], ha="center", va="bottom",
                    fontsize=8, color=GROUP_COLOR[g], fontweight="bold",
                    transform=ax.get_xaxis_transform())
            if start > 0:
                ax.axvline(start - 0.5, color="lightgray", lw=1, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS_SHORT[m] for m in metric_order],
                           rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.7)
        ax.axhline(0.7, color="gray", lw=0.8, linestyle=":", alpha=0.7)
        ax.text(-0.6, 0.505, "ρ = 0.5", fontsize=7.5, color="gray", va="bottom")
        ax.text(-0.6, 0.705, "ρ = 0.7", fontsize=7.5, color="gray", va="bottom")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
        ax.set_title(
            "Inter-judge agreement (Spearman ρ) across all 9 metrics and 3 comparison pairs\n"
            "Error bars = 95% bootstrap CI  ·  n = 3,504 trials",
            fontsize=10, pad=12,
        )
        fig.tight_layout()
        out = PLOT_DIR / "fig5_rho_grouped_barplot.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — Scatter plots: Qwen-QO vs Qwen-QAS (the clean B-C comparison)
#          3×3 grid, one per metric, trial-level scores with ρ annotation
# ══════════════════════════════════════════════════════════════════════════
def plot_scatter_9metrics(merged):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 3, figsize=(11, 10))
        axes = axes.flat

        jitter = 0.12

        for ax, metric in zip(axes, METRICS):
            x = merged[f"{metric}_qo"].values.astype(float)
            y = merged[f"{metric}_qas"].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            xj = x + np.random.uniform(-jitter, jitter, len(x))
            yj = y + np.random.uniform(-jitter, jitter, len(y))

            rho, _ = stats.spearmanr(x, y)
            group  = metric_group(metric)
            color  = GROUP_COLOR[group]

            ax.scatter(xj, yj, s=4, alpha=0.25, color=color, linewidths=0)
            # diagonal reference
            ax.plot([1, 5], [1, 5], color="black", lw=0.8, ls="--", alpha=0.5)

            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_aspect("equal")
            ax.set_title(METRIC_LABELS_SHORT[metric], fontsize=9.5, color=color, fontweight="bold")
            ax.text(0.97, 0.05, f"ρ = {rho:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, fontweight="bold", color=color)

        fig.supxlabel("Qwen-QO score  (without support images)", fontsize=10, y=0.01)
        fig.supylabel("Qwen-QAS score  (with support images)", fontsize=10, x=0.01)
        fig.suptitle(
            "Trial-level score agreement: Qwen-QO vs Qwen-QAS (same model, ± support images)\n"
            "Each point = one trial  ·  Dashed line = perfect agreement  ·  n = 3,504",
            fontsize=10, y=1.01,
        )
        # Group legend
        patches = [mpatches.Patch(color=GROUP_COLOR[g], label=GROUP_LABEL[g])
                   for g in ["robust", "sensitive", "intermediate", "other"]]
        fig.legend(handles=patches, loc="lower center", ncol=2,
                   fontsize=8.5, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
        fig.tight_layout()
        out = PLOT_DIR / "fig6_scatter_9metrics_BvsC.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Bland-Altman for the B-C comparison, 4 key metrics
#          Shows bias (mean difference) and limits of agreement
# ══════════════════════════════════════════════════════════════════════════
def plot_bland_altman(merged):
    key_metrics = [
        "discriminativeness", "textual_groundedness",
        "hallucination_free", "comprehensibility",
    ]
    key_titles = {
        "discriminativeness":   "Local Discriminativeness  (robust)",
        "textual_groundedness": "Textual Groundedness  (sensitive)",
        "hallucination_free":   "Hallucination Free  (sensitive)",
        "comprehensibility":    "Comprehensibility  (robust)",
    }

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flat

        for ax, metric in zip(axes, key_metrics):
            b  = merged[f"{metric}_qo"].values.astype(float)
            c  = merged[f"{metric}_qas"].values.astype(float)
            mask = np.isfinite(b) & np.isfinite(c)
            b, c = b[mask], c[mask]

            means = (b + c) / 2
            diffs = b - c          # Qwen-QO minus Qwen-QAS

            mean_diff = np.mean(diffs)
            std_diff  = np.std(diffs, ddof=1)
            loa_lo = mean_diff - 1.96 * std_diff
            loa_hi = mean_diff + 1.96 * std_diff

            group = metric_group(metric)
            color = GROUP_COLOR[group]

            ax.scatter(means, diffs, s=5, alpha=0.2, color=color, linewidths=0)
            ax.axhline(mean_diff, color="black",  lw=1.5, label=f"Mean diff = {mean_diff:+.3f}")
            ax.axhline(loa_lo,   color="crimson", lw=1,   ls="--",
                       label=f"−1.96 SD = {loa_lo:.3f}")
            ax.axhline(loa_hi,   color="crimson", lw=1,   ls="--",
                       label=f"+1.96 SD = {loa_hi:.3f}")
            ax.axhline(0, color="gray", lw=0.7, ls=":")

            ax.set_xlim(0.5, 5.5)
            ax.set_xlabel("Mean of Qwen-QO and Qwen-QAS scores", fontsize=9)
            ax.set_ylabel("Qwen-QO − Qwen-QAS", fontsize=9)
            ax.set_title(key_titles[metric], fontsize=10, color=color, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

        fig.suptitle(
            "Bland-Altman agreement plots: Qwen-QO vs Qwen-QAS (same model, ± support images)\n"
            "Dashed red lines = 95% limits of agreement (mean ± 1.96 SD)  ·  n = 3,504",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        out = PLOT_DIR / "fig7_bland_altman_BvsC.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# FIG 8 — Mean score differences (calibration offsets) with 95% CI
#          Shows that GPT is systematically more strict than Qwen
# ══════════════════════════════════════════════════════════════════════════
def plot_calibration_offsets(summary_df):
    with plt.rc_context(STYLE):
        sub = summary_df[summary_df["subset"] == "all"].copy()

        pairs = ["A-B", "A-C", "B-C"]
        pair_labels = {
            "A-B": "GPT-QO − Qwen-QO",
            "A-C": "GPT-QO − Qwen-QAS",
            "B-C": "Qwen-QO − Qwen-QAS",
        }
        pair_colors = {"A-B": "#546E7A", "A-C": "#78909C", "B-C": "#1565C0"}
        pair_markers = {"A-B": "s", "A-C": "D", "B-C": "o"}

        # Same metric order as fig5
        order_groups = ["robust", "sensitive", "other", "intermediate"]
        bc = sub[sub["pair"] == "B-C"].set_index("metric")["mean_diff"]
        metric_order = []
        for g in order_groups:
            ms = [m for m in METRICS if metric_group(m) == g]
            ms.sort(key=lambda m: -bc.get(m, 0))
            metric_order.extend(ms)

        y = np.arange(len(metric_order))
        offsets = [-0.22, 0, 0.22]

        fig, ax = plt.subplots(figsize=(9, 7))

        for i, pair in enumerate(pairs):
            pdf = sub[sub["pair"] == pair].set_index("metric")
            diffs  = [pdf.loc[m, "mean_diff"]  if m in pdf.index else np.nan for m in metric_order]
            ci_lo  = [pdf.loc[m, "ci_lower"]   if m in pdf.index else np.nan for m in metric_order]
            ci_hi  = [pdf.loc[m, "ci_upper"]   if m in pdf.index else np.nan for m in metric_order]
            yerr_lo = [d - lo for d, lo in zip(diffs, ci_lo)]
            yerr_hi = [hi - d  for d, hi in zip(diffs, ci_hi)]
            ax.errorbar(
                diffs, y + offsets[i],
                xerr=[yerr_lo, yerr_hi],
                fmt=pair_markers[pair],
                color=pair_colors[pair],
                markersize=7, capsize=4, elinewidth=1.2,
                label=pair_labels[pair], alpha=0.9,
            )

        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([METRIC_LABELS_SHORT[m] for m in metric_order], fontsize=9)

        # Group separators
        group_spans = {}
        for m in metric_order:
            g = metric_group(m)
            idx = metric_order.index(m)
            if g not in group_spans:
                group_spans[g] = [idx, idx]
            else:
                group_spans[g][1] = idx
        for g, (start, end) in group_spans.items():
            mid = (start + end) / 2
            ax.text(ax.get_xlim()[1] + 0.02, mid, GROUP_LABEL[g],
                    ha="left", va="center", fontsize=7.5,
                    color=GROUP_COLOR[g], fontweight="bold",
                    transform=ax.get_yaxis_transform())
            if start > 0:
                ax.axhline(start - 0.5, color="lightgray", lw=1, ls="--")

        ax.set_xlabel("Mean score difference  (1–5 scale)   negative = first judge scores lower")
        ax.set_title(
            "Calibration offsets between judges: mean score difference with 95% CI\n"
            "n = 3,504 trials  ·  Positive = left judge scores higher",
            fontsize=10, pad=10,
        )
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        out = PLOT_DIR / "fig8_calibration_offsets.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import random
    np.random.seed(42)
    random.seed(42)

    print("Loading data…")
    merged   = load_merged()
    spearman = pd.read_csv(SPEARMAN_CSV)
    summary  = pd.read_csv(BASE / "summary_stats.csv")

    print(f"Merged trials: {len(merged)}")

    print("\nGenerating fig5 — ρ grouped bar chart…")
    plot_rho_barplot(spearman)

    print("Generating fig6 — 9-metric scatter panel (B vs C)…")
    plot_scatter_9metrics(merged)

    print("Generating fig7 — Bland-Altman (B vs C, 4 key metrics)…")
    plot_bland_altman(merged)

    print("Generating fig8 — calibration offsets…")
    plot_calibration_offsets(summary)

    print("\nAll done.")
