from __future__ import annotations

from collections import defaultdict
from itertools import chain, combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon


DIMENSION_FIELDS = (
    "textual_groundedness",
    "hallucination_free",
    "concept_counting",
    "comprehensibility",
    "conciseness",
    "specificity",
    "discriminativeness",
    "instruction_following",
    "logical_coherence",
)

DIMENSION_LABELS = (
    "Groundedness",
    "Hallucination\nFree",
    "Concept\nCounting",
    "Comprehen-\nsibility",
    "Conciseness",
    "Specificity",
    "Discrimina-\ntiveness",
    "Instruction\nFollowing",
    "Logical\nCoherence",
)

# Fieldname helpers
_SCORE_FIELDS = (*DIMENSION_FIELDS, "overall_score")
_SCORE_FIELDS_WITH_SE = tuple(chain.from_iterable((f, f"{f}_se") for f in _SCORE_FIELDS))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _se(values: List[float]) -> float:
    if len(values) < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}" if not math.isnan(value) else ""


def _mean_score_table(rows: List[Dict[str, str]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Mean ± SE for all 9 dimensions + overall_score, grouped by group_keys.

    Each metric yields two columns: ``<field>`` (mean) and ``<field>_se``.
    """
    grouped: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(
        lambda: {field: [] for field in _SCORE_FIELDS}
    )

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        for field in _SCORE_FIELDS:
            value = _safe_float(row.get(field))
            if not math.isnan(value):
                grouped[key][field].append(value)

    output: List[Dict[str, Any]] = []
    for key, metric_values in sorted(grouped.items()):
        out_row: Dict[str, Any] = {group_keys[i]: v for i, v in enumerate(key)}
        overall_vals = metric_values["overall_score"]
        out_row["judged_trials"] = len(overall_vals)
        for field in _SCORE_FIELDS:
            vals = metric_values[field]
            out_row[field] = _fmt(mean(vals)) if vals else ""
            out_row[f"{field}_se"] = _fmt(_se(vals)) if len(vals) >= 2 else ""
        output.append(out_row)
    return output


def _dimension_prompt_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Long-format table: (prompt_type, dimension) → mean ± SE."""
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        prompt_type = row["prompt_type"]
        for dimension in DIMENSION_FIELDS:
            value = _safe_float(row.get(dimension))
            if not math.isnan(value):
                grouped[(prompt_type, dimension)].append(value)

    output: List[Dict[str, Any]] = []
    for (prompt_type, dimension), values in sorted(grouped.items()):
        output.append(
            {
                "prompt_type": prompt_type,
                "dimension": dimension,
                "judged_trials": len(values),
                "mean_score": _fmt(mean(values)),
                "se_score": _fmt(_se(values)) if len(values) >= 2 else "",
            }
        )
    return output


def _spearman_bootstrap(
    x: List[float],
    y: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Spearman rho + bootstrap 95 % CI. Returns (rho, ci_low, ci_high)."""
    if len(x) < 3:
        return math.nan, math.nan, math.nan
    rho_val, _ = spearmanr(x, y)
    rng = np.random.default_rng(42)
    xa, ya = np.array(x), np.array(y)
    n = len(xa)
    boot_rhos: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xi, yi = xa[idx], ya[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            continue
        r, _ = spearmanr(xi, yi)
        boot_rhos.append(float(r))
    if not boot_rhos:
        return float(rho_val), math.nan, math.nan
    alpha = (1 - confidence) / 2
    return float(rho_val), float(np.quantile(boot_rhos, alpha)), float(np.quantile(boot_rhos, 1 - alpha))


def _spearman_table(
    judge_rows: List[Dict[str, str]],
    trial_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Table B4: Spearman correlation (+ bootstrap CI) between each judge dimension and
    binary classifier accuracy, broken down by prompt_type × dimension.

    Requires trial_results.csv rows for the join on
    (model/source_model, dataset, prompt_type, config_n/k/q, run_id, query_index_within_episode).
    """
    correct_lookup: Dict[Tuple[str, ...], int] = {}
    for row in trial_rows:
        key = (
            row.get("model", ""),
            row.get("dataset", ""),
            row.get("prompt_type", ""),
            str(row.get("config_n", "")),
            str(row.get("config_k", "")),
            str(row.get("config_q", "")),
            str(row.get("run_id", "")),
            str(row.get("query_index_within_episode", "")),
        )
        correct_val = row.get("correct", "")
        if correct_val in ("0", "1"):
            correct_lookup[key] = int(correct_val)

    # (prompt_type, dimension) → ([dim_scores], [correct_values])
    buckets: Dict[Tuple[str, str], Tuple[List[float], List[int]]] = defaultdict(lambda: ([], []))
    for row in judge_rows:
        join_key = (
            row.get("source_model", ""),
            row.get("dataset", ""),
            row.get("prompt_type", ""),
            str(row.get("config_n", "")),
            str(row.get("config_k", "")),
            str(row.get("config_q", "")),
            str(row.get("run_id", "")),
            str(row.get("query_index_within_episode", "")),
        )
        correct = correct_lookup.get(join_key)
        if correct is None:
            continue
        prompt_type = row.get("prompt_type", "")
        for dim in DIMENSION_FIELDS:
            score = _safe_float(row.get(dim))
            if not math.isnan(score):
                buckets[(prompt_type, dim)][0].append(score)
                buckets[(prompt_type, dim)][1].append(correct)

    output: List[Dict[str, Any]] = []
    for (prompt_type, dimension), (scores, corrects) in sorted(buckets.items()):
        rho, ci_low, ci_high = _spearman_bootstrap(scores, corrects)
        output.append(
            {
                "prompt_type": prompt_type,
                "dimension": dimension,
                "n_matched": len(scores),
                "spearman_rho": _fmt(rho),
                "ci95_low": _fmt(ci_low),
                "ci95_high": _fmt(ci_high),
            }
        )
    return output


def _plot_bar(
    *,
    title: str,
    path: Path,
    x_labels: List[str],
    series: Dict[str, List[float]],
    ylabel: str = "Mean Score",
    y_max: float = 5.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(x_labels) * 1.5), 5))
    series_names = list(series.keys())
    width = 0.8 / max(len(series_names), 1)
    positions = list(range(len(x_labels)))

    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2) * width
        plt.bar(
            [pos + offset for pos in positions],
            series[series_name],
            width=width,
            label=series_name,
        )

    plt.xticks(positions, x_labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.ylim(0.0, y_max)
    plt.title(title)
    if len(series_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_radar(
    *,
    title: str,
    path: Path,
    categories: Sequence[str],
    series: Dict[str, List[float]],
    y_max: float = 5.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, y_max)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=7)
    ax.yaxis.set_tick_params(labelsize=7)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, values) in enumerate(series.items()):
        data = values + values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, data, linewidth=1.5, linestyle="solid", label=label, color=color)
        ax.fill(angles, data, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)
    ax.set_title(title, size=10, pad=18)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _pairwise_wilcoxon_trial_overall(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row["source_run_name"],
            row["dataset"],
            row["config_n"],
            row["config_k"],
            row["config_q"],
            row["run_id"],
            row["query_index_within_episode"],
        )
        grouped[key][row["prompt_type"]] = _safe_float(row["overall_score"])

    prompt_types = sorted({row["prompt_type"] for row in rows})
    output: List[Dict[str, Any]] = []
    for left, right in combinations(prompt_types, 2):
        left_values: List[float] = []
        right_values: List[float] = []
        for score_map in grouped.values():
            if left not in score_map or right not in score_map:
                continue
            ls, rs = score_map[left], score_map[right]
            if math.isnan(ls) or math.isnan(rs):
                continue
            left_values.append(ls)
            right_values.append(rs)

        if not left_values:
            continue

        statistic = ""
        p_value = ""
        if any(a != b for a, b in zip(left_values, right_values)):
            test = wilcoxon(left_values, right_values, zero_method="wilcox", alternative="two-sided")
            statistic = f"{test.statistic:.4f}"
            p_value = f"{test.pvalue:.6f}"

        output.append(
            {
                "prompt_a": left,
                "prompt_b": right,
                "matched_trials": len(left_values),
                "mean_overall_a": _fmt(mean(left_values)),
                "mean_overall_b": _fmt(mean(right_values)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    return output


def _friedman_trial_overall(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt_types = sorted({row["prompt_type"] for row in rows})
    if len(prompt_types) < 3:
        return []

    grouped: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            row["source_run_name"],
            row["dataset"],
            row["config_n"],
            row["config_k"],
            row["config_q"],
            row["run_id"],
            row["query_index_within_episode"],
        )
        grouped[key][row["prompt_type"]] = _safe_float(row["overall_score"])

    matched = [
        ps for ps in grouped.values()
        if all(p in ps and not math.isnan(ps[p]) for p in prompt_types)
    ]
    if len(matched) < 2:
        return []

    samples = [[ps[p] for ps in matched] for p in prompt_types]
    test = friedmanchisquare(*samples)
    return [
        {
            "matched_trials": len(matched),
            "prompt_types": ",".join(prompt_types),
            "friedman_statistic": f"{test.statistic:.6f}",
            "p_value": f"{test.pvalue:.6f}",
        }
    ]


def _short_label(label: str, max_len: int = 22) -> str:
    if "/" in label:
        label = label.split("/")[-1]
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def _build_grouped_dim_scores(
    rows: List[Dict[str, str]],
    x_key: str,
    series_key: str,
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    """Group per-dimension scores by (x_key, series_key).

    Returns (x_labels, {series_val: {x_val: {dimension: mean_score}}}).
    """
    x_labels = sorted({row[x_key] for row in rows if row.get(x_key)})
    series_vals = sorted({row[series_key] for row in rows if row.get(series_key)})

    buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        s: {x: {d: [] for d in DIMENSION_FIELDS} for x in x_labels}
        for s in series_vals
    }
    for row in rows:
        x = row.get(x_key, "")
        s = row.get(series_key, "")
        if not x or not s:
            continue
        for dim in DIMENSION_FIELDS:
            v = _safe_float(row.get(dim))
            if not math.isnan(v):
                buckets[s][x][dim].append(v)

    series_data: Dict[str, Dict[str, Dict[str, float]]] = {
        s: {
            x: {dim: mean(vals) if vals else 0.0 for dim, vals in dim_scores.items()}
            for x, dim_scores in x_data.items()
        }
        for s, x_data in buckets.items()
    }
    return x_labels, series_data


def _plot_metric_dashboard(
    *,
    title: str,
    path: Path,
    x_labels: List[str],
    series_data: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """3×3 grid of grouped bar charts — one panel per dimension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_names = list(series_data.keys())
    n_series = max(len(series_names), 1)
    bar_width = 0.75 / n_series
    positions = list(range(len(x_labels)))
    short_x = [_short_label(x) for x in x_labels]

    for panel_idx, (dim, dim_label) in enumerate(zip(DIMENSION_FIELDS, DIMENSION_LABELS)):
        ax = axes[panel_idx // 3][panel_idx % 3]
        for s_idx, s_name in enumerate(series_names):
            offset = (s_idx - (n_series - 1) / 2) * bar_width
            values = [series_data[s_name].get(x, {}).get(dim, 0.0) for x in x_labels]
            ax.bar(
                [p + offset for p in positions],
                values,
                width=bar_width,
                label=_short_label(s_name),
                color=colors[s_idx % len(colors)],
                alpha=0.85,
            )
        ax.set_title(dim_label.replace("\n", " "), fontsize=9, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(short_x, rotation=25, ha="right", fontsize=7)
        ax.set_ylim(0, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("Mean Score", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.axhline(y=3.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(n_series, 6),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.00),
        frameon=True,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_judge_run_directory(run_dir: Path) -> Dict[str, Path]:
    judge_results_path = run_dir / "judge_results.csv"
    if not judge_results_path.exists():
        raise FileNotFoundError(f"Missing judge results file: {judge_results_path}")

    rows = _read_csv(judge_results_path)
    scored_rows = [row for row in rows if not math.isnan(_safe_float(row.get("overall_score")))]

    analysis_dir = run_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"
    stats_dir = analysis_dir / "stats"

    # ── Tables ──────────────────────────────────────────────────────────────

    # B1: 9 metrics × prompt_type (aggregated over models and datasets)
    b1_table = _mean_score_table(scored_rows, ["prompt_type"])
    _write_csv(
        tables_dir / "B1_metrics_by_prompt.csv",
        ["prompt_type", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b1_table,
    )

    # B2: 9 metrics × source model (aggregated over datasets and conditions)
    b2_table = _mean_score_table(scored_rows, ["source_model"])
    _write_csv(
        tables_dir / "B2_metrics_by_model.csv",
        ["source_model", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b2_table,
    )

    # B3: 9 metrics × dataset (aggregated over models and conditions)
    b3_table = _mean_score_table(scored_rows, ["dataset"])
    _write_csv(
        tables_dir / "B3_metrics_by_dataset.csv",
        ["dataset", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b3_table,
    )

    # Existing cross tables (kept for backward compat, now include SE)
    dataset_prompt_table = _mean_score_table(scored_rows, ["dataset", "prompt_type"])
    _write_csv(
        tables_dir / "mean_scores_by_dataset_and_prompt.csv",
        ["dataset", "prompt_type", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        dataset_prompt_table,
    )

    dimension_table = _dimension_prompt_table(scored_rows)
    _write_csv(
        tables_dir / "mean_scores_by_dimension_and_prompt.csv",
        ["prompt_type", "dimension", "judged_trials", "mean_score", "se_score"],
        dimension_table,
    )

    # B4: Spearman correlation (dim score × classifier accuracy) with bootstrap CI
    b4_table: List[Dict[str, Any]] = []
    trial_results_path = run_dir.parent.parent / "trial_results.csv"
    b4_available = trial_results_path.exists()
    if b4_available:
        trial_rows = _read_csv(trial_results_path)
        b4_table = _spearman_table(scored_rows, trial_rows)
        if b4_table:
            _write_csv(
                tables_dir / "B4_correlation_metric_accuracy.csv",
                ["prompt_type", "dimension", "n_matched", "spearman_rho", "ci95_low", "ci95_high"],
                b4_table,
            )

    # B5: 9 metrics × (source_model × prompt_type)
    b5_table = _mean_score_table(scored_rows, ["source_model", "prompt_type"])
    _write_csv(
        tables_dir / "B5_metrics_by_model_and_prompt.csv",
        ["source_model", "prompt_type", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b5_table,
    )

    # B6: 9 metrics × (source_model × dataset)
    b6_table = _mean_score_table(scored_rows, ["source_model", "dataset"])
    _write_csv(
        tables_dir / "B6_metrics_by_model_and_dataset.csv",
        ["source_model", "dataset", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b6_table,
    )

    # B7: 9 metrics × (source_model × dataset × prompt_type) — most granular
    b7_table = _mean_score_table(scored_rows, ["source_model", "dataset", "prompt_type"])
    _write_csv(
        tables_dir / "B7_metrics_by_model_dataset_and_prompt.csv",
        ["source_model", "dataset", "prompt_type", "judged_trials", *_SCORE_FIELDS_WITH_SE],
        b7_table,
    )

    # ── Statistical tests ────────────────────────────────────────────────────

    wilcoxon_rows = _pairwise_wilcoxon_trial_overall(scored_rows)
    friedman_rows = _friedman_trial_overall(scored_rows)

    if wilcoxon_rows:
        _write_csv(
            stats_dir / "pairwise_wilcoxon_trial_overall_score.csv",
            ["prompt_a", "prompt_b", "matched_trials", "mean_overall_a", "mean_overall_b",
             "wilcoxon_statistic", "p_value"],
            wilcoxon_rows,
        )
    if friedman_rows:
        _write_csv(
            stats_dir / "friedman_trial_overall_score.csv",
            ["matched_trials", "prompt_types", "friedman_statistic", "p_value"],
            friedman_rows,
        )

    # ── Plots ────────────────────────────────────────────────────────────────

    _plot_bar(
        title="Judge Mean Overall Score by Prompt Type",
        path=plots_dir / "overall_score_by_prompt.png",
        x_labels=[row["prompt_type"] for row in b1_table],
        series={"overall_score": [_safe_float(row["overall_score"]) for row in b1_table]},
    )

    dimension_labels = list(DIMENSION_FIELDS)
    prompt_types = sorted({row["prompt_type"] for row in dimension_table})
    dimension_series: Dict[str, List[float]] = {}
    for prompt_type in prompt_types:
        dimension_series[prompt_type] = [
            next(
                (_safe_float(r["mean_score"]) for r in dimension_table
                 if r["prompt_type"] == prompt_type and r["dimension"] == dim),
                0.0,
            )
            for dim in dimension_labels
        ]
    _plot_bar(
        title="Judge Mean Score by Dimension and Prompt Type",
        path=plots_dir / "score_by_dimension_and_prompt.png",
        x_labels=dimension_labels,
        series=dimension_series,
    )

    dataset_names = sorted({row["dataset"] for row in dataset_prompt_table})
    dataset_series: Dict[str, List[float]] = {}
    for prompt_type in sorted({row["prompt_type"] for row in dataset_prompt_table}):
        dataset_series[prompt_type] = [
            next(
                (_safe_float(row["overall_score"]) for row in dataset_prompt_table
                 if row["dataset"] == ds and row["prompt_type"] == prompt_type),
                0.0,
            )
            for ds in dataset_names
        ]
    _plot_bar(
        title="Judge Mean Overall Score by Dataset and Prompt Type",
        path=plots_dir / "overall_score_by_dataset_and_prompt.png",
        x_labels=dataset_names,
        series=dataset_series,
    )

    # ── Radar charts ─────────────────────────────────────────────────────────
    radar_categories = list(DIMENSION_LABELS)

    # Radar 1 (Fig. principal): all prompt types overlaid, 9 dimensions as axes
    prompt_radar_series: Dict[str, List[float]] = {}
    for prompt_type in prompt_types:
        prompt_radar_series[prompt_type] = [
            next(
                (_safe_float(r["mean_score"]) for r in dimension_table
                 if r["prompt_type"] == prompt_type and r["dimension"] == dim),
                0.0,
            )
            for dim in DIMENSION_FIELDS
        ]
    if len(prompt_radar_series) >= 2:
        _plot_radar(
            title="Judge Scores by Dimension and Prompt Type",
            path=plots_dir / "radar_by_prompt.png",
            categories=radar_categories,
            series=prompt_radar_series,
        )

    # Radar 2: all source models overlaid (only if >1 model)
    source_models = sorted({row["source_model"] for row in scored_rows if row.get("source_model")})
    if len(source_models) >= 2:
        model_table = _mean_score_table(scored_rows, ["source_model"])
        model_radar_series: Dict[str, List[float]] = {
            row["source_model"]: [_safe_float(row.get(dim, 0.0)) for dim in DIMENSION_FIELDS]
            for row in model_table
        }
        _plot_radar(
            title="Judge Scores by Dimension and Source Model",
            path=plots_dir / "radar_by_model.png",
            categories=radar_categories,
            series=model_radar_series,
        )

    # Radar 3 (optional, appendix): per prompt type, one series per dataset
    radar_dataset_dir = plots_dir / "radar_by_dataset"
    prompt_dataset_table = _mean_score_table(scored_rows, ["prompt_type", "dataset"])
    all_datasets = sorted({row["dataset"] for row in scored_rows})
    radar_dataset_files: List[str] = []
    for prompt_type in sorted({row["prompt_type"] for row in scored_rows}):
        dataset_radar_series: Dict[str, List[float]] = {}
        for dataset_name in all_datasets:
            matched = next(
                (r for r in prompt_dataset_table
                 if r["prompt_type"] == prompt_type and r["dataset"] == dataset_name),
                None,
            )
            if matched:
                dataset_radar_series[dataset_name] = [_safe_float(matched.get(dim, 0.0)) for dim in DIMENSION_FIELDS]
        if len(dataset_radar_series) >= 2:
            radar_file = radar_dataset_dir / f"radar_{prompt_type}.png"
            _plot_radar(
                title=f"Judge Scores by Dimension and Dataset — {prompt_type}",
                path=radar_file,
                categories=radar_categories,
                series=dataset_radar_series,
            )
            radar_dataset_files.append(f"radar_by_dataset/radar_{prompt_type}.png")

    # ── 9-metric dashboards ──────────────────────────────────────────────────
    dashboard_files: List[str] = []

    x_labels_cond, series_cond = _build_grouped_dim_scores(scored_rows, "prompt_type", "source_model")
    if x_labels_cond and series_cond:
        _plot_metric_dashboard(
            title="All Metrics by Experimental Condition  (series = model)",
            path=plots_dir / "dashboard_by_condition.png",
            x_labels=x_labels_cond,
            series_data=series_cond,
        )
        dashboard_files.append("plots/dashboard_by_condition.png")

    x_labels_model, series_model = _build_grouped_dim_scores(scored_rows, "source_model", "prompt_type")
    if x_labels_model and series_model:
        _plot_metric_dashboard(
            title="All Metrics by Model  (series = experimental condition)",
            path=plots_dir / "dashboard_by_model.png",
            x_labels=x_labels_model,
            series_data=series_model,
        )
        dashboard_files.append("plots/dashboard_by_model.png")

    x_labels_ds, series_ds = _build_grouped_dim_scores(scored_rows, "dataset", "prompt_type")
    if x_labels_ds and series_ds:
        _plot_metric_dashboard(
            title="All Metrics by Dataset  (series = experimental condition)",
            path=plots_dir / "dashboard_by_dataset.png",
            x_labels=x_labels_ds,
            series_data=series_ds,
        )
        dashboard_files.append("plots/dashboard_by_dataset.png")

    # ── Report ───────────────────────────────────────────────────────────────
    report_path = analysis_dir / "report.md"
    best_prompt = ""
    if b1_table:
        best_prompt = max(b1_table, key=lambda r: _safe_float(r["overall_score"]))["prompt_type"]

    report_lines = [
        "# Judge Run Report",
        "",
        f"- Judge results file: `{judge_results_path.name}`",
        f"- Prompt types analyzed: {', '.join(sorted({row['prompt_type'] for row in scored_rows}))}" if scored_rows else "- Prompt types analyzed: n/a",
        f"- Best prompt by mean overall judge score: `{best_prompt}`" if best_prompt else "- Best prompt by mean overall judge score: n/a",
        "",
        "## Paper tables (mean ± SE per cell)",
        "- `tables/B1_metrics_by_prompt.csv` — 9 metrics × prompt type (aggregated over models and datasets)",
        "- `tables/B2_metrics_by_model.csv` — 9 metrics × source model (aggregated over datasets and conditions)",
        "- `tables/B3_metrics_by_dataset.csv` — 9 metrics × dataset (aggregated over models and conditions)",
        "- `tables/B4_correlation_metric_accuracy.csv` — Spearman ρ (+ bootstrap 95 % CI) between each dimension and binary accuracy, by prompt type" if b4_table else "- Table B4 not generated (trial_results.csv not found at expected path)",
        "",
        "## Appendix tables (mean ± SE per cell)",
        "- `tables/B5_metrics_by_model_and_prompt.csv` — 9 metrics × (source model × prompt type)",
        "- `tables/B6_metrics_by_model_and_dataset.csv` — 9 metrics × (source model × dataset)",
        "- `tables/B7_metrics_by_model_dataset_and_prompt.csv` — 9 metrics × (source model × dataset × prompt type), most granular",
        "",
        "## Supporting tables",
        "- `tables/mean_scores_by_dataset_and_prompt.csv` — 9 metrics × (dataset × prompt type), mean ± SE",
        "- `tables/mean_scores_by_dimension_and_prompt.csv` — long format: (prompt type, dimension) → mean ± SE",
        "",
        "## Statistical tests",
        "- `stats/pairwise_wilcoxon_trial_overall_score.csv`" if wilcoxon_rows else "- Pairwise Wilcoxon not generated",
        "- `stats/friedman_trial_overall_score.csv`" if friedman_rows else "- Friedman test not generated",
        "",
        "## 9-metric dashboards (no averaging — all dimensions shown)",
        *[f"- `{f}`" for f in dashboard_files],
        "",
        "## Per-dimension detail plots",
        "- `plots/score_by_dimension_and_prompt.png`",
        "- `plots/radar_by_prompt.png` — all prompt types overlaid; axes = 9 dimensions" if len(prompt_radar_series) >= 2 else "- Radar by prompt: not generated (need ≥2 prompt types)",
        "- `plots/radar_by_model.png` — all source models overlaid" if len(source_models) >= 2 else "- Radar by model: not generated (need ≥2 models)",
        *[f"- `plots/{f}`" for f in radar_dataset_files],
        "",
        "## Summary plots (overall score)",
        "- `plots/overall_score_by_prompt.png`",
        "- `plots/overall_score_by_dataset_and_prompt.png`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "analysis_dir": analysis_dir,
        "report_path": report_path,
    }
