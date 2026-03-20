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
from scipy.stats import binomtest, friedmanchisquare, wilcoxon


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


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    ci = binomtest(successes, total).proportion_ci(confidence_level=confidence, method="exact")
    return (ci.low, ci.high)


def _accuracy_table(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        key = tuple(row[key_name] for key_name in group_keys)
        grouped[key]["correct"] += int(row["correct"])
        grouped[key]["total"] += 1

    output: List[Dict[str, Any]] = []
    for key, counts in sorted(grouped.items()):
        correct = counts["correct"]
        total = counts["total"]
        low, high = _wilson_interval(correct, total)
        row = {group_keys[idx]: value for idx, value in enumerate(key)}
        row.update(
            {
                "correct": correct,
                "total": total,
                "accuracy": f"{correct / total:.4f}" if total else "",
                "ci95_low": f"{low:.4f}" if not math.isnan(low) else "",
                "ci95_high": f"{high:.4f}" if not math.isnan(high) else "",
            }
        )
        output.append(row)
    return output


def _plot_bar(
    *,
    title: str,
    path: Path,
    x_labels: List[str],
    series: Dict[str, List[float]],
    ylabel: str = "Accuracy",
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
    plt.ylim(0.0, 1.0)
    plt.title(title)
    if len(series_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _pairwise_mcnemar(trial_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, int]] = defaultdict(dict)
    for row in trial_rows:
        key = (
            row["dataset"],
            row["config_n"],
            row["config_k"],
            row["config_q"],
            row["run_id"],
            row["query_index_within_episode"],
        )
        grouped[key][row["prompt_type"]] = int(row["correct"])

    prompt_types = sorted({row["prompt_type"] for row in trial_rows})
    output: List[Dict[str, Any]] = []
    for left, right in combinations(prompt_types, 2):
        both_correct = 0
        both_wrong = 0
        left_only = 0
        right_only = 0

        for prompt_map in grouped.values():
            if left not in prompt_map or right not in prompt_map:
                continue
            left_value = prompt_map[left]
            right_value = prompt_map[right]
            if left_value == 1 and right_value == 1:
                both_correct += 1
            elif left_value == 0 and right_value == 0:
                both_wrong += 1
            elif left_value == 1 and right_value == 0:
                left_only += 1
            else:
                right_only += 1

        discordant = left_only + right_only
        p_value = ""
        better_prompt = ""
        if discordant > 0:
            p_value = f"{binomtest(left_only, discordant, p=0.5).pvalue:.6f}"
            if left_only > right_only:
                better_prompt = left
            elif right_only > left_only:
                better_prompt = right
            else:
                better_prompt = "tie"

        output.append(
            {
                "prompt_a": left,
                "prompt_b": right,
                "matched_trials": both_correct + both_wrong + discordant,
                "a_correct_b_wrong": left_only,
                "b_correct_a_wrong": right_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "exact_p_value": p_value,
                "better_prompt": better_prompt,
            }
        )
    return output


def _pairwise_wilcoxon(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in run_rows:
        key = (
            row["dataset"],
            row["config_n"],
            row["config_k"],
            row["config_q"],
            row["run_id"],
        )
        grouped[key][row["prompt_type"]] = _safe_float(row["accuracy"])

    prompt_types = sorted({row["prompt_type"] for row in run_rows})
    output: List[Dict[str, Any]] = []

    for left, right in combinations(prompt_types, 2):
        left_values: List[float] = []
        right_values: List[float] = []
        for prompt_map in grouped.values():
            if left not in prompt_map or right not in prompt_map:
                continue
            left_values.append(prompt_map[left])
            right_values.append(prompt_map[right])

        if not left_values:
            continue

        nonzero_diffs = [a != b for a, b in zip(left_values, right_values)]
        p_value = ""
        statistic = ""
        if any(nonzero_diffs):
            stat = wilcoxon(left_values, right_values, zero_method="wilcox", alternative="two-sided")
            p_value = f"{stat.pvalue:.6f}"
            statistic = f"{stat.statistic:.4f}"

        output.append(
            {
                "prompt_a": left,
                "prompt_b": right,
                "matched_runs": len(left_values),
                "mean_accuracy_a": f"{mean(left_values):.4f}",
                "mean_accuracy_b": f"{mean(right_values):.4f}",
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    return output


def _friedman(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_types = sorted({row["prompt_type"] for row in run_rows})
    if len(prompt_types) < 3:
        return []

    grouped: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in run_rows:
        key = (
            row["dataset"],
            row["config_n"],
            row["config_k"],
            row["config_q"],
            row["run_id"],
        )
        grouped[key][row["prompt_type"]] = _safe_float(row["accuracy"])

    matched = [prompt_map for prompt_map in grouped.values() if all(prompt in prompt_map for prompt in prompt_types)]
    if len(matched) < 2:
        return []

    samples = [[prompt_map[prompt] for prompt_map in matched] for prompt in prompt_types]
    stat = friedmanchisquare(*samples)
    return [
        {
            "matched_runs": len(matched),
            "prompt_types": ",".join(prompt_types),
            "friedman_statistic": f"{stat.statistic:.6f}",
            "p_value": f"{stat.pvalue:.6f}",
        }
    ]


def analyze_run_directory(run_dir: Path) -> Dict[str, Path]:
    trial_results_path = run_dir / "trial_results.csv"
    run_accuracy_path = run_dir / "run_accuracy_long.csv"
    if not trial_results_path.exists():
        raise FileNotFoundError(f"Missing trial results file: {trial_results_path}")
    if not run_accuracy_path.exists():
        raise FileNotFoundError(f"Missing run accuracy file: {run_accuracy_path}")

    trial_rows = _read_csv(trial_results_path)
    run_rows = _read_csv(run_accuracy_path)

    analysis_dir = run_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"
    stats_dir = analysis_dir / "stats"

    overall_table = _accuracy_table(trial_rows, ["prompt_type"])
    dataset_table = _accuracy_table(trial_rows, ["dataset", "prompt_type"])
    config_table = _accuracy_table(trial_rows, ["config_n", "config_k", "config_q", "prompt_type"])

    _write_csv(
        tables_dir / "overall_accuracy_by_prompt.csv",
        ["prompt_type", "correct", "total", "accuracy", "ci95_low", "ci95_high"],
        overall_table,
    )
    _write_csv(
        tables_dir / "accuracy_by_dataset_and_prompt.csv",
        ["dataset", "prompt_type", "correct", "total", "accuracy", "ci95_low", "ci95_high"],
        dataset_table,
    )
    _write_csv(
        tables_dir / "accuracy_by_config_and_prompt.csv",
        ["config_n", "config_k", "config_q", "prompt_type", "correct", "total", "accuracy", "ci95_low", "ci95_high"],
        config_table,
    )

    mcnemar_rows = _pairwise_mcnemar(trial_rows)
    wilcoxon_rows = _pairwise_wilcoxon(run_rows)
    friedman_rows = _friedman(run_rows)

    if mcnemar_rows:
        _write_csv(
            stats_dir / "pairwise_mcnemar.csv",
            [
                "prompt_a",
                "prompt_b",
                "matched_trials",
                "a_correct_b_wrong",
                "b_correct_a_wrong",
                "both_correct",
                "both_wrong",
                "exact_p_value",
                "better_prompt",
            ],
            mcnemar_rows,
        )
    if wilcoxon_rows:
        _write_csv(
            stats_dir / "pairwise_wilcoxon_run_accuracy.csv",
            [
                "prompt_a",
                "prompt_b",
                "matched_runs",
                "mean_accuracy_a",
                "mean_accuracy_b",
                "wilcoxon_statistic",
                "p_value",
            ],
            wilcoxon_rows,
        )
    if friedman_rows:
        _write_csv(
            stats_dir / "friedman_run_accuracy.csv",
            ["matched_runs", "prompt_types", "friedman_statistic", "p_value"],
            friedman_rows,
        )

    overall_plot_series = {"accuracy": [_safe_float(row["accuracy"]) for row in overall_table]}
    _plot_bar(
        title="Overall Accuracy by Prompt Type",
        path=plots_dir / "overall_accuracy_by_prompt.png",
        x_labels=[row["prompt_type"] for row in overall_table],
        series=overall_plot_series,
    )

    dataset_names = sorted({row["dataset"] for row in dataset_table})
    prompt_types = sorted({row["prompt_type"] for row in dataset_table})
    dataset_series: Dict[str, List[float]] = {}
    for prompt_type in prompt_types:
        dataset_series[prompt_type] = []
        for dataset_name in dataset_names:
            matched = next(
                (
                    _safe_float(row["accuracy"])
                    for row in dataset_table
                    if row["dataset"] == dataset_name and row["prompt_type"] == prompt_type
                ),
                0.0,
            )
            dataset_series[prompt_type].append(matched)
    _plot_bar(
        title="Accuracy by Dataset and Prompt Type",
        path=plots_dir / "accuracy_by_dataset_and_prompt.png",
        x_labels=dataset_names,
        series=dataset_series,
    )

    config_labels = sorted(
        {
            f"N={row['config_n']},K={row['config_k']},Q={row['config_q']}"
            for row in config_table
        }
    )
    config_series: Dict[str, List[float]] = {}
    for prompt_type in prompt_types:
        config_series[prompt_type] = []
        for label in config_labels:
            matched = 0.0
            for row in config_table:
                row_label = f"N={row['config_n']},K={row['config_k']},Q={row['config_q']}"
                if row["prompt_type"] == prompt_type and row_label == label:
                    matched = _safe_float(row["accuracy"])
                    break
            config_series[prompt_type].append(matched)
    _plot_bar(
        title="Accuracy by Few-Shot Configuration",
        path=plots_dir / "accuracy_by_config_and_prompt.png",
        x_labels=config_labels,
        series=config_series,
    )

    report_path = analysis_dir / "report.md"
    best_prompt = ""
    if overall_table:
        best_prompt = max(overall_table, key=lambda row: _safe_float(row["accuracy"]))["prompt_type"]

    report_lines = [
        "# OpenRouter Experiment Report",
        "",
        f"- Trials file: `{trial_results_path.name}`",
        f"- Run summaries: `{run_accuracy_path.name}`",
        f"- Prompt types analyzed: {', '.join(prompt_types)}",
        f"- Best overall prompt by raw accuracy: `{best_prompt}`" if best_prompt else "- Best overall prompt by raw accuracy: n/a",
        "",
        "## Generated tables",
        "- `tables/overall_accuracy_by_prompt.csv`",
        "- `tables/accuracy_by_dataset_and_prompt.csv`",
        "- `tables/accuracy_by_config_and_prompt.csv`",
        "",
        "## Generated plots",
        "- `plots/overall_accuracy_by_prompt.png`",
        "- `plots/accuracy_by_dataset_and_prompt.png`",
        "- `plots/accuracy_by_config_and_prompt.png`",
    ]

    if mcnemar_rows or wilcoxon_rows or friedman_rows:
        report_lines.extend(
            [
                "",
                "## Statistical tests",
                "- `stats/pairwise_mcnemar.csv`" if mcnemar_rows else "- Pairwise McNemar not generated",
                "- `stats/pairwise_wilcoxon_run_accuracy.csv`" if wilcoxon_rows else "- Pairwise Wilcoxon not generated",
                "- `stats/friedman_run_accuracy.csv`" if friedman_rows else "- Friedman test not generated",
            ]
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "analysis_dir": analysis_dir,
        "report_path": report_path,
    }
