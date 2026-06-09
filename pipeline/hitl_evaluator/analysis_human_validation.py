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

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating plots ─────────────────────────────────────────────")
    plot_scatter(merged)
    plot_kappa_heatmap(iaa, items)
    plot_boxplot_diff_by_condition(merged)

    print(f"\n✓ All outputs in: {RESULTS_DIR}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
