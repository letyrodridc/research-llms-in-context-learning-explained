from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon


DIMENSION_FIELDS = (
    "visual_grounding",
    "discriminative_support",
    "inferential_coherence",
    "clarity",
    "format_compliance",
)


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


def _mean_score_table(rows: List[Dict[str, str]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(
        lambda: {field: [] for field in (*DIMENSION_FIELDS, "overall_score")}
    )

    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        for field in (*DIMENSION_FIELDS, "overall_score"):
            value = _safe_float(row.get(field))
            if not math.isnan(value):
                grouped[key][field].append(value)

    output: List[Dict[str, Any]] = []
    for key, metric_values in sorted(grouped.items()):
        row = {group_keys[idx]: value for idx, value in enumerate(key)}
        overall_values = metric_values["overall_score"]
        row["judged_trials"] = len(overall_values)
        for field in DIMENSION_FIELDS:
            values = metric_values[field]
            row[field] = f"{mean(values):.4f}" if values else ""
        row["overall_score"] = f"{mean(overall_values):.4f}" if overall_values else ""
        output.append(row)
    return output


def _dimension_prompt_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
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
                "mean_score": f"{mean(values):.4f}",
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


def _pairwise_wilcoxon_trial_overall(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, float]] = defaultdict(dict)
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
            left_score = score_map[left]
            right_score = score_map[right]
            if math.isnan(left_score) or math.isnan(right_score):
                continue
            left_values.append(left_score)
            right_values.append(right_score)

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
                "mean_overall_a": f"{mean(left_values):.4f}",
                "mean_overall_b": f"{mean(right_values):.4f}",
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    return output


def _friedman_trial_overall(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    prompt_types = sorted({row["prompt_type"] for row in rows})
    if len(prompt_types) < 3:
        return []

    grouped: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, float]] = defaultdict(dict)
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
        prompt_scores
        for prompt_scores in grouped.values()
        if all(prompt in prompt_scores and not math.isnan(prompt_scores[prompt]) for prompt in prompt_types)
    ]
    if len(matched) < 2:
        return []

    samples = [[prompt_scores[prompt] for prompt_scores in matched] for prompt in prompt_types]
    test = friedmanchisquare(*samples)
    return [
        {
            "matched_trials": len(matched),
            "prompt_types": ",".join(prompt_types),
            "friedman_statistic": f"{test.statistic:.6f}",
            "p_value": f"{test.pvalue:.6f}",
        }
    ]


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

    overall_table = _mean_score_table(scored_rows, ["prompt_type"])
    dataset_table = _mean_score_table(scored_rows, ["dataset", "prompt_type"])
    dimension_table = _dimension_prompt_table(scored_rows)

    _write_csv(
        tables_dir / "overall_mean_scores_by_prompt.csv",
        ["prompt_type", "judged_trials", *DIMENSION_FIELDS, "overall_score"],
        overall_table,
    )
    _write_csv(
        tables_dir / "mean_scores_by_dataset_and_prompt.csv",
        ["dataset", "prompt_type", "judged_trials", *DIMENSION_FIELDS, "overall_score"],
        dataset_table,
    )
    _write_csv(
        tables_dir / "mean_scores_by_dimension_and_prompt.csv",
        ["prompt_type", "dimension", "judged_trials", "mean_score"],
        dimension_table,
    )

    wilcoxon_rows = _pairwise_wilcoxon_trial_overall(scored_rows)
    friedman_rows = _friedman_trial_overall(scored_rows)

    if wilcoxon_rows:
        _write_csv(
            stats_dir / "pairwise_wilcoxon_trial_overall_score.csv",
            [
                "prompt_a",
                "prompt_b",
                "matched_trials",
                "mean_overall_a",
                "mean_overall_b",
                "wilcoxon_statistic",
                "p_value",
            ],
            wilcoxon_rows,
        )
    if friedman_rows:
        _write_csv(
            stats_dir / "friedman_trial_overall_score.csv",
            ["matched_trials", "prompt_types", "friedman_statistic", "p_value"],
            friedman_rows,
        )

    _plot_bar(
        title="Judge Mean Overall Score by Prompt Type",
        path=plots_dir / "overall_score_by_prompt.png",
        x_labels=[row["prompt_type"] for row in overall_table],
        series={"overall_score": [_safe_float(row["overall_score"]) for row in overall_table]},
    )

    dimension_labels = list(DIMENSION_FIELDS)
    prompt_types = sorted({row["prompt_type"] for row in dimension_table})
    dimension_series: Dict[str, List[float]] = {}
    for prompt_type in prompt_types:
        dimension_series[prompt_type] = []
        for dimension in dimension_labels:
            matched_score = next(
                (
                    _safe_float(row["mean_score"])
                    for row in dimension_table
                    if row["prompt_type"] == prompt_type and row["dimension"] == dimension
                ),
                0.0,
            )
            dimension_series[prompt_type].append(matched_score)
    _plot_bar(
        title="Judge Mean Score by Dimension and Prompt Type",
        path=plots_dir / "score_by_dimension_and_prompt.png",
        x_labels=dimension_labels,
        series=dimension_series,
    )

    dataset_names = sorted({row["dataset"] for row in dataset_table})
    dataset_series: Dict[str, List[float]] = {}
    for prompt_type in sorted({row["prompt_type"] for row in dataset_table}):
        dataset_series[prompt_type] = []
        for dataset_name in dataset_names:
            matched_score = next(
                (
                    _safe_float(row["overall_score"])
                    for row in dataset_table
                    if row["dataset"] == dataset_name and row["prompt_type"] == prompt_type
                ),
                0.0,
            )
            dataset_series[prompt_type].append(matched_score)
    _plot_bar(
        title="Judge Mean Overall Score by Dataset and Prompt Type",
        path=plots_dir / "overall_score_by_dataset_and_prompt.png",
        x_labels=dataset_names,
        series=dataset_series,
    )

    report_path = analysis_dir / "report.md"
    best_prompt = ""
    if overall_table:
        best_prompt = max(overall_table, key=lambda row: _safe_float(row["overall_score"]))["prompt_type"]

    report_lines = [
        "# Judge Run Report",
        "",
        f"- Judge results file: `{judge_results_path.name}`",
        f"- Prompt types analyzed: {', '.join(sorted({row['prompt_type'] for row in scored_rows}))}" if scored_rows else "- Prompt types analyzed: n/a",
        f"- Best prompt by mean overall judge score: `{best_prompt}`" if best_prompt else "- Best prompt by mean overall judge score: n/a",
        "",
        "## Generated tables",
        "- `tables/overall_mean_scores_by_prompt.csv`",
        "- `tables/mean_scores_by_dataset_and_prompt.csv`",
        "- `tables/mean_scores_by_dimension_and_prompt.csv`",
        "",
        "## Generated plots",
        "- `plots/overall_score_by_prompt.png`",
        "- `plots/score_by_dimension_and_prompt.png`",
        "- `plots/overall_score_by_dataset_and_prompt.png`",
        "",
        "## Statistical tests",
        "- `stats/pairwise_wilcoxon_trial_overall_score.csv`" if wilcoxon_rows else "- Pairwise Wilcoxon not generated",
        "- `stats/friedman_trial_overall_score.csv`" if friedman_rows else "- Friedman test not generated",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "analysis_dir": analysis_dir,
        "report_path": report_path,
    }
