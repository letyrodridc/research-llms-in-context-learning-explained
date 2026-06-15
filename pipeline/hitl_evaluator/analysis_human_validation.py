"""
Human validation analysis: inter-annotator agreement + judge alignment.

Usage (from repo root, after all 3 annotators have exported and pushed their CSVs):
    python pipeline/hitl_evaluator/analysis_human_validation.py

Inputs expected in pipeline/hitl_evaluator/:
    annotations_carmen_final.csv
    annotations_leticia_final.csv
    annotations_nico_final.csv

Also reads judge scores from the main run directory.

Outputs go to pipeline/hitl_evaluator/results/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HITL_DIR   = os.path.dirname(__file__)
RUN_DIR    = os.path.join(REPO_ROOT, "pipeline", "openrouter_runs",
                          "full_experiment_single__20260427_213833")
JUDGE_CSV  = os.path.join(RUN_DIR, "judge_outputs", "openai-gpt-5-mini",
                          "judge_results.csv")
RESULTS_DIR = os.path.join(HITL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── metric mapping ────────────────────────────────────────────────────────────
METRICS = {
    "TG": "textual_groundedness",
    "HF": "hallucination_free",
    "CC": "concept_counting",
    "CP": "comprehensibility",
    "Cn": "conciseness",
    "S":  "specificity",
    "LD": "discriminativeness",
    "IF": "instruction_following",
    "LC": "logical_coherence",
}
HUMAN_COLS  = {k: f"human_{k}" for k in METRICS}
METRIC_KEYS = list(METRICS.keys())

JOIN_KEY = ["model", "dataset", "prompt_type",
            "config_n", "config_k", "run_id", "query_index_within_episode"]
JOIN_KEY_JUDGE = ["source_model", "dataset", "prompt_type",
                  "config_n", "config_k", "run_id", "query_index_within_episode"]


# ── helpers ───────────────────────────────────────────────────────────────────

def weighted_kappa_linear(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's κ with linear weights for ordinal scale 1-5."""
    labels = [1, 2, 3, 4, 5]
    try:
        return cohen_kappa_score(y1, y2, labels=labels, weights="linear")
    except Exception:
        return float("nan")


def load_annotator_csvs() -> pd.DataFrame:
    """Load and concatenate the 3 final annotator CSVs."""
    frames = []
    for name in ["carmen", "leticia", "nico"]:
        path = os.path.join(HITL_DIR, f"annotations_{name}_final.csv")
        if not os.path.exists(path):
            print(f"⚠  Not found: {path} — skipping")
            continue
        df = pd.read_csv(path)
        df["annotator"] = name
        frames.append(df)

    if not frames:
        print("ERROR: No annotator CSVs found. Run the annotation app and export first.")
        sys.exit(1)

    all_ann = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(all_ann)} annotation rows from {len(frames)} annotators")
    return all_ann


def verify_coverage(all_ann: pd.DataFrame):
    """Check each item_id appears in exactly 2 annotator rows."""
    counts = all_ann.groupby("item_id")["annotator"].nunique()
    bad = counts[counts != 2]
    if len(bad):
        print(f"⚠  WARNING: {len(bad)} items without exactly 2 annotators:")
        print(bad[bad != 2].head(10))
    else:
        print(f"✓ All {len(counts)} items annotated by exactly 2 annotators")
    return counts


def build_item_table(all_ann: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per item_id with annotator scores side by side.
    Columns: item_id, ann1_<metric>, ann2_<metric>, mean_<metric>, annotator_pair
    """
    rows = []
    for item_id, grp in all_ann.groupby("item_id"):
        grp = grp.sort_values("annotator").reset_index(drop=True)
        if len(grp) < 2:
            continue
        row = {"item_id": item_id, "annotator_pair": "_".join(grp["annotator"].tolist())}
        # Carry over join key columns from first annotator's row
        for col in JOIN_KEY + ["dataset", "prompt_type", "condition", "model_short", "stratum"]:
            if col in grp.columns:
                row[col] = grp.iloc[0][col]

        for mk in METRIC_KEYS:
            hcol = HUMAN_COLS[mk]
            if hcol not in grp.columns:
                continue
            v1 = pd.to_numeric(grp.iloc[0][hcol], errors="coerce")
            v2 = pd.to_numeric(grp.iloc[1][hcol], errors="coerce")
            row[f"ann1_{mk}"] = v1
            row[f"ann2_{mk}"] = v2
            row[f"mean_{mk}"] = np.nanmean([v1, v2])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_iaa(items: pd.DataFrame) -> pd.DataFrame:
    """Inter-annotator agreement: weighted κ per metric."""
    results = []
    for mk in METRIC_KEYS:
        col1, col2 = f"ann1_{mk}", f"ann2_{mk}"
        if col1 not in items.columns or col2 not in items.columns:
            continue
        valid = items[[col1, col2]].dropna()
        if len(valid) < 10:
            kappa = float("nan")
        else:
            kappa = weighted_kappa_linear(
                valid[col1].astype(int).values,
                valid[col2].astype(int).values,
            )
        results.append({
            "metric": mk,
            "n_pairs": len(valid),
            "kappa_linear": round(kappa, 3),
        })
    return pd.DataFrame(results)


def compute_judge_alignment(items: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    """
    Join items table with judge scores (by item join key), then compute
    Spearman ρ and MAE between mean_human and judge score per metric.
    """
    # Map model column name
    judge_renamed = judge.rename(columns={"source_model": "model"})

    merged = items.merge(
        judge_renamed[JOIN_KEY + list(METRICS.values())],
        on=JOIN_KEY,
        how="left",
    )
    n_matched = merged[list(METRICS.values())[0]].notna().sum()
    print(f"Matched {n_matched}/{len(merged)} items with judge scores")

    results = []
    for mk, jcol in METRICS.items():
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            continue
        valid = merged[[mean_col, jcol]].dropna()
        if len(valid) < 5:
            rho, mae = float("nan"), float("nan")
        else:
            rho, _pval = stats.spearmanr(valid[mean_col], valid[jcol])
            mae = (valid[mean_col] - valid[jcol]).abs().mean()
        results.append({
            "metric": mk,
            "n_items": len(valid),
            "spearman_rho": round(rho, 3) if not np.isnan(rho) else float("nan"),
            "mae": round(mae, 3) if not np.isnan(mae) else float("nan"),
        })
    return pd.DataFrame(results), merged


def build_latex_table(iaa: pd.DataFrame, align: pd.DataFrame) -> str:
    """Produce a LaTeX table with κ, ρ, MAE per metric."""
    combined = iaa.merge(align, on="metric")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Human–Judge Alignment and Inter-Annotator Agreement}",
        r"\label{tab:hitl_alignment}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{$\kappa_w$} & \textbf{Spearman $\rho$} & \textbf{MAE} \\",
        r"\midrule",
    ]
    for _, row in combined.iterrows():
        kappa = f"{row['kappa_linear']:.3f}" if not np.isnan(row["kappa_linear"]) else "--"
        rho   = f"{row['spearman_rho']:.3f}" if not np.isnan(row["spearman_rho"]) else "--"
        mae   = f"{row['mae']:.3f}" if not np.isnan(row["mae"]) else "--"
        lines.append(f"{row['metric']} & {kappa} & {rho} & {mae} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plot_scatter(merged: pd.DataFrame):
    """3×3 scatter plot: judge score vs mean human score per metric."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    axes = axes.flatten()
    for i, (mk, jcol) in enumerate(METRICS.items()):
        ax = axes[i]
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            ax.set_visible(False)
            continue
        valid = merged[[mean_col, jcol]].dropna()
        ax.scatter(valid[jcol], valid[mean_col], alpha=0.4, s=20, color="#0d6efd")
        ax.plot([1, 5], [1, 5], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
        ax.set_xlabel("Judge score"); ax.set_ylabel("Mean human score")
        ax.set_title(mk, fontweight="bold")
        if len(valid) >= 5:
            rho, _ = stats.spearmanr(valid[jcol], valid[mean_col])
            ax.text(0.05, 0.93, f"ρ={rho:.2f}, n={len(valid)}",
                    transform=ax.transAxes, fontsize=8)
    plt.suptitle("Judge vs. Human Scores (per metric)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "scatter_judge_vs_human.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_kappa_heatmap(iaa: pd.DataFrame, items: pd.DataFrame):
    """Heatmap of κ by metric and annotator pair."""
    pairs = sorted(items["annotator_pair"].unique()) if "annotator_pair" in items.columns else []
    if not pairs:
        return

    kappa_data = {}
    for pair in pairs:
        subset = items[items["annotator_pair"] == pair]
        kappas = {}
        for mk in METRIC_KEYS:
            c1, c2 = f"ann1_{mk}", f"ann2_{mk}"
            if c1 not in subset.columns:
                continue
            valid = subset[[c1, c2]].dropna()
            if len(valid) >= 5:
                kappas[mk] = round(weighted_kappa_linear(
                    valid[c1].astype(int).values,
                    valid[c2].astype(int).values,
                ), 3)
            else:
                kappas[mk] = float("nan")
        kappa_data[pair] = kappas

    df_heat = pd.DataFrame(kappa_data, index=METRIC_KEYS).T
    fig, ax = plt.subplots(figsize=(10, 4))
    mask = df_heat.isna()
    sns.heatmap(df_heat.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-0.2, vmax=1.0, ax=ax, mask=mask, linewidths=0.5)
    ax.set_title("Weighted κ (linear) by Annotator Pair × Metric", fontweight="bold")
    ax.set_xlabel("Metric"); ax.set_ylabel("Annotator pair")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heatmap_kappa_by_annotator_pair.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_annotator_calibration(all_ann: pd.DataFrame):
    """Heatmap of mean score per annotator × metric — reveals calibration differences."""
    HUMAN_COLS_LIST = list(HUMAN_COLS.values())
    ann_clean = all_ann[all_ann[HUMAN_COLS_LIST[0]].notna()].copy()
    for c in HUMAN_COLS_LIST:
        ann_clean[c] = pd.to_numeric(ann_clean[c], errors="coerce")

    rows = {}
    for name in ["carmen", "leticia", "nico"]:
        sub = ann_clean[ann_clean["annotator"] == name]
        if len(sub) == 0:
            continue
        rows[name] = {mk: sub[HUMAN_COLS[mk]].mean() for mk in METRIC_KEYS}
    if not rows:
        return

    df = pd.DataFrame(rows, index=METRIC_KEYS).T
    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=1, vmax=5, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Mean score (1–5)"})
    ax.set_title("Mean Score per Annotator × Metric  (calibration check)", fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Annotator")

    # Add overall mean per annotator on the right
    overall = df.mean(axis=1)
    for i, (name, val) in enumerate(overall.items()):
        ax.text(len(METRIC_KEYS) + 0.15, i + 0.5, f"μ={val:.2f}",
                va="center", fontsize=9, color="#333")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heatmap_annotator_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_bias_barchart(merged: pd.DataFrame):
    """Horizontal bar chart of mean(judge − human) per metric with ±1 SD."""
    biases, errors, labels = [], [], []
    for mk, jcol in METRICS.items():
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            continue
        diff = (merged[jcol] - merged[mean_col]).dropna()
        biases.append(diff.mean())
        errors.append(diff.std())
        labels.append(mk)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#d9534f" if b < 0 else "#5cb85c" for b in biases]
    y = np.arange(len(labels))
    ax.barh(y, biases, xerr=errors, color=colors, alpha=0.8,
            error_kw={"ecolor": "#555", "capsize": 4, "lw": 1.2})
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean (Judge score − Human mean)", fontsize=10)
    ax.set_title("Systematic Bias: Judge vs. Humans per Metric\n(red = judge underscores, green = judge overscores)",
                 fontweight="bold")
    for i, (b, e) in enumerate(zip(biases, errors)):
        ax.text(b + (0.05 if b >= 0 else -0.05), i,
                f"{b:+.2f}", va="center",
                ha="left" if b >= 0 else "right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bias_barchart_judge_vs_human.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def plot_score_distributions(merged: pd.DataFrame, all_ann: pd.DataFrame):
    """
    Violin plot per metric showing score distributions for each of the 3 annotators
    and the judge side by side — reveals both spread and calibration differences.
    """
    HUMAN_COLS_LIST = list(HUMAN_COLS.values())
    ann_clean = all_ann[all_ann[HUMAN_COLS_LIST[0]].notna()].copy()
    for c in HUMAN_COLS_LIST:
        ann_clean[c] = pd.to_numeric(ann_clean[c], errors="coerce")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    annotator_colors = {"carmen": "#e07b54", "leticia": "#5b9bd5", "nico": "#70ad47", "judge": "#9b59b6"}

    for i, (mk, jcol) in enumerate(METRICS.items()):
        ax = axes[i]
        hcol = HUMAN_COLS[mk]
        data, tick_labels, colors = [], [], []

        for name in ["carmen", "leticia", "nico"]:
            sub = ann_clean[ann_clean["annotator"] == name][hcol].dropna()
            # Only include items that are in the merged (validated) set
            data.append(sub.values)
            tick_labels.append(name.capitalize())
            colors.append(annotator_colors[name])

        # Judge scores
        if jcol in merged.columns:
            data.append(merged[jcol].dropna().values)
            tick_labels.append("Judge")
            colors.append(annotator_colors["judge"])

        parts = ax.violinplot(data, positions=range(len(data)), showmedians=True,
                              showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(mk, fontweight="bold")
        ax.set_ylabel("Score")
        ax.axhline(4.0, color="gray", lw=0.7, ls=":")

    plt.suptitle("Score Distributions per Metric: Annotators vs. Judge",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "violin_score_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def export_tables(all_ann: pd.DataFrame, items: pd.DataFrame,
                  merged: pd.DataFrame, iaa: pd.DataFrame, align: pd.DataFrame):
    """Export all intermediate analysis tables as CSVs."""
    HUMAN_COLS_LIST = list(HUMAN_COLS.values())
    ann_clean = all_ann[all_ann[HUMAN_COLS_LIST[0]].notna()].copy()
    for c in HUMAN_COLS_LIST:
        ann_clean[c] = pd.to_numeric(ann_clean[c], errors="coerce")

    # ── 1. Calibration: mean score per annotator × metric ─────────────────────
    rows_cal = []
    for name in ["carmen", "leticia", "nico"]:
        sub = ann_clean[ann_clean["annotator"] == name]
        row = {"annotator": name}
        for mk in METRIC_KEYS:
            row[mk] = round(sub[HUMAN_COLS[mk]].mean(), 3)
        row["overall_mean"] = round(sub[HUMAN_COLS_LIST].mean().mean(), 3)
        row["n_items"] = len(sub)
        rows_cal.append(row)
    df_cal = pd.DataFrame(rows_cal)
    df_cal.to_csv(os.path.join(RESULTS_DIR, "calibration_by_annotator.csv"), index=False)
    print("✓ calibration_by_annotator.csv")

    # ── 2. Systematic bias: judge − human per metric ──────────────────────────
    rows_bias = []
    for mk, jcol in METRICS.items():
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            continue
        diff = (merged[jcol] - merged[mean_col]).dropna()
        rows_bias.append({
            "metric": mk,
            "n": len(diff),
            "mean_bias": round(diff.mean(), 4),
            "std_bias": round(diff.std(), 4),
            "median_bias": round(diff.median(), 4),
            "min_bias": round(diff.min(), 2),
            "max_bias": round(diff.max(), 2),
            "pct_judge_higher": round((diff > 0).mean() * 100, 1),
            "pct_judge_lower": round((diff < 0).mean() * 100, 1),
            "pct_equal": round((diff == 0).mean() * 100, 1),
        })
    pd.DataFrame(rows_bias).to_csv(
        os.path.join(RESULTS_DIR, "bias_judge_vs_human.csv"), index=False)
    print("✓ bias_judge_vs_human.csv")

    # ── 3. Mean score comparison: human_mean vs judge per metric ─────────────
    rows_comp = []
    for mk, jcol in METRICS.items():
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            continue
        hm = merged[mean_col].mean()
        jm = merged[jcol].mean()
        rows_comp.append({
            "metric": mk,
            "human_mean": round(hm, 3),
            "judge_mean": round(jm, 3),
            "diff_judge_minus_human": round(jm - hm, 3),
            "human_std": round(merged[mean_col].std(), 3),
            "judge_std": round(merged[jcol].std(), 3),
            "human_median": round(merged[mean_col].median(), 3),
            "judge_median": round(merged[jcol].median(), 3),
        })
    pd.DataFrame(rows_comp).to_csv(
        os.path.join(RESULTS_DIR, "score_comparison_human_vs_judge.csv"), index=False)
    print("✓ score_comparison_human_vs_judge.csv")

    # ── 4. κ per annotator pair × metric ──────────────────────────────────────
    pairs = sorted(items["annotator_pair"].unique()) if "annotator_pair" in items.columns else []
    rows_kp = []
    for pair in pairs:
        subset = items[items["annotator_pair"] == pair]
        for mk in METRIC_KEYS:
            c1, c2 = f"ann1_{mk}", f"ann2_{mk}"
            if c1 not in subset.columns:
                continue
            valid = subset[[c1, c2]].dropna()
            kappa = float("nan")
            if len(valid) >= 5:
                kappa = round(weighted_kappa_linear(
                    valid[c1].astype(int).values,
                    valid[c2].astype(int).values), 3)
            rows_kp.append({
                "annotator_pair": pair,
                "metric": mk,
                "n_pairs": len(valid),
                "kappa_linear": kappa,
                "ann1_mean": round(valid[c1].mean(), 3) if len(valid) else float("nan"),
                "ann2_mean": round(valid[c2].mean(), 3) if len(valid) else float("nan"),
                "mean_diff_abs": round((valid[c1] - valid[c2]).abs().mean(), 3) if len(valid) else float("nan"),
            })
    pd.DataFrame(rows_kp).to_csv(
        os.path.join(RESULTS_DIR, "kappa_by_pair_and_metric.csv"), index=False)
    print("✓ kappa_by_pair_and_metric.csv")

    # ── 5. Bias per condition × metric ────────────────────────────────────────
    if "condition" in merged.columns:
        rows_cond = []
        for cond in sorted(merged["condition"].dropna().unique()):
            sub_cond = merged[merged["condition"] == cond]
            for mk, jcol in METRICS.items():
                mean_col = f"mean_{mk}"
                if mean_col not in sub_cond.columns or jcol not in sub_cond.columns:
                    continue
                diff = (sub_cond[jcol] - sub_cond[mean_col]).dropna()
                if len(diff) == 0:
                    continue
                rows_cond.append({
                    "condition": cond,
                    "metric": mk,
                    "n": len(diff),
                    "mean_bias": round(diff.mean(), 4),
                    "std_bias": round(diff.std(), 4),
                    "median_bias": round(diff.median(), 4),
                })
        pd.DataFrame(rows_cond).to_csv(
            os.path.join(RESULTS_DIR, "bias_by_condition_and_metric.csv"), index=False)
        print("✓ bias_by_condition_and_metric.csv")

    # ── 6. Full summary: all metrics in one table ─────────────────────────────
    df_full = iaa.merge(align, on="metric")
    # Add bias and score comparison
    df_bias = pd.DataFrame(rows_bias)[["metric", "mean_bias", "std_bias",
                                        "pct_judge_higher", "pct_judge_lower"]]
    df_scores = pd.DataFrame(rows_comp)[["metric", "human_mean", "judge_mean"]]
    df_full = df_full.merge(df_bias, on="metric").merge(df_scores, on="metric")
    df_full.to_csv(os.path.join(RESULTS_DIR, "full_summary.csv"), index=False)
    print("✓ full_summary.csv")


def plot_boxplot_diff_by_condition(merged: pd.DataFrame):
    """Boxplot of judge–human difference by condition (E2-E5) for each metric."""
    if "condition" not in merged.columns:
        return
    conditions = sorted(merged["condition"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    for i, (mk, jcol) in enumerate(METRICS.items()):
        ax = axes[i]
        mean_col = f"mean_{mk}"
        if mean_col not in merged.columns or jcol not in merged.columns:
            ax.set_visible(False)
            continue
        data_by_cond = []
        labels = []
        for cond in conditions:
            sub = merged[merged["condition"] == cond][[mean_col, jcol]].dropna()
            if len(sub):
                data_by_cond.append((sub[jcol] - sub[mean_col]).values)
                labels.append(cond)
        if data_by_cond:
            ax.boxplot(data_by_cond, tick_labels=labels, showfliers=False)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_ylabel("Judge − Human")
            ax.set_title(mk, fontweight="bold")
    plt.suptitle("Judge − Mean Human Score by Condition", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "boxplot_diff_by_condition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


def main():
    print("═" * 60)
    print("  HITL Validation Analysis")
    print("═" * 60)

    # ── load human annotations ────────────────────────────────────────────────
    all_ann = load_annotator_csvs()

    human_metric_cols = list(HUMAN_COLS.values())
    missing_human = [c for c in human_metric_cols if c not in all_ann.columns]
    if missing_human:
        print(f"ERROR: Missing columns in annotator CSVs: {missing_human}")
        print("Make sure annotators exported the final CSV after annotating.")
        sys.exit(1)

    # Drop rows where no human scores filled (not yet annotated)
    all_ann = all_ann[all_ann[human_metric_cols[0]].notna() &
                      (all_ann[human_metric_cols[0]] != "")].copy()
    for col in human_metric_cols:
        all_ann[col] = pd.to_numeric(all_ann[col], errors="coerce")

    verify_coverage(all_ann)

    # ── build item table ──────────────────────────────────────────────────────
    items = build_item_table(all_ann)
    print(f"Item table: {len(items)} items with 2-annotator pairs")

    # ── load judge CSV ────────────────────────────────────────────────────────
    if not os.path.exists(JUDGE_CSV):
        print(f"ERROR: Judge CSV not found at {JUDGE_CSV}")
        sys.exit(1)
    judge = pd.read_csv(JUDGE_CSV)

    # ── inter-annotator agreement ─────────────────────────────────────────────
    print("\n── Inter-Annotator Agreement (Cohen's κ weighted linear) ────────")
    iaa = compute_iaa(items)
    print(iaa.to_string(index=False))

    # ── judge alignment ───────────────────────────────────────────────────────
    print("\n── Judge–Human Alignment ────────────────────────────────────────")
    align, merged = compute_judge_alignment(items, judge)
    print(align.to_string(index=False))

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = build_latex_table(iaa, align)
    latex_path = os.path.join(RESULTS_DIR, "table_alignment.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\n✓ LaTeX table: {latex_path}")
    print("\n" + latex)

    # ── save CSV ──────────────────────────────────────────────────────────────
    combined = iaa.merge(align, on="metric")
    results_csv = os.path.join(RESULTS_DIR, "alignment_results.csv")
    combined.to_csv(results_csv, index=False)
    item_csv = os.path.join(RESULTS_DIR, "items_with_scores.csv")
    merged.to_csv(item_csv, index=False)
    print(f"✓ Results CSV: {results_csv}")
    print(f"✓ Full item table: {item_csv}")

    # ── export all intermediate tables ────────────────────────────────────────
    print("\n── Exporting intermediate analysis tables ───────────────────────")
    export_tables(all_ann, items, merged, iaa, align)

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────────────────")
    plot_scatter(merged)
    plot_kappa_heatmap(iaa, items)
    plot_boxplot_diff_by_condition(merged)
    plot_annotator_calibration(all_ann)
    plot_bias_barchart(merged)
    plot_score_distributions(merged, all_ann)

    print(f"\n✓ All outputs in: {RESULTS_DIR}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
