from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import csv
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, chi2, chi2_contingency, fisher_exact

ALPHA = 0.05
MATCHED_GROUP_FIELDS = {"prompt_type", "model"}
MATCH_BASE_FIELDS = [
    "dataset",
    "prompt_type",
    "model",
    "config_n",
    "config_k",
    "config_q",
    "run_id",
    "query_index_within_episode",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _slugify(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip())
    return safe.strip("-").lower() or "value"


def _natural_sort_key(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))


def _wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    ci = binomtest(successes, total).proportion_ci(confidence_level=confidence, method="wilson")
    return (ci.low, ci.high)


def _standard_error(successes: int, total: int) -> float:
    if total == 0:
        return math.nan
    proportion = successes / total
    return math.sqrt(proportion * (1.0 - proportion) / total)


def _aggregate_accuracy(rows: List[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key]["correct"] += _safe_int(row["correct"])
        grouped[key]["total"] += 1

    output: List[Dict[str, Any]] = []
    for key, counts in sorted(grouped.items()):
        correct = counts["correct"]
        total = counts["total"]
        accuracy = correct / total if total else math.nan
        low, high = _wilson_interval(correct, total)
        standard_error = _standard_error(correct, total)
        row = {group_fields[idx]: value for idx, value in enumerate(key)}
        row.update(
            {
                "correct": correct,
                "total": total,
                "accuracy": f"{accuracy:.4f}" if total else "",
                "standard_error": f"{standard_error:.6f}" if total else "",
                "ci95_low": f"{low:.4f}" if not math.isnan(low) else "",
                "ci95_high": f"{high:.4f}" if not math.isnan(high) else "",
            }
        )
        output.append(row)
    return output


def _holm_adjust(p_values: List[float]) -> List[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for idx, (original_index, p_value) in enumerate(indexed):
        candidate = min(1.0, (total - idx) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _pairwise_exact_mcnemar(group_a: str, group_b: str, values: List[Tuple[int, int]]) -> Dict[str, Any]:
    a_only = sum(1 for left, right in values if left == 1 and right == 0)
    b_only = sum(1 for left, right in values if left == 0 and right == 1)
    both_correct = sum(1 for left, right in values if left == 1 and right == 1)
    both_wrong = sum(1 for left, right in values if left == 0 and right == 0)
    discordant = a_only + b_only
    p_value = binomtest(a_only, discordant, p=0.5).pvalue if discordant else 1.0
    return {
        "group_a": group_a,
        "group_b": group_b,
        "matched_units": len(values),
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "p_value": p_value,
    }


def _matched_values(rows: List[Dict[str, Any]], group_field: str) -> Tuple[List[str], Dict[Tuple[Any, ...], Dict[str, int]]]:
    groups = sorted({str(row[group_field]) for row in rows}, key=_natural_sort_key)
    grouped: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(dict)
    match_fields = [field for field in MATCH_BASE_FIELDS if field != group_field]
    for row in rows:
        key = tuple(row[field] for field in match_fields)
        grouped[key][str(row[group_field])] = _safe_int(row["correct"])
    grouped = {
        key: value_map
        for key, value_map in grouped.items()
        if all(group in value_map for group in groups)
    }
    return groups, grouped


def _cochrans_q(groups: List[str], grouped: Mapping[Tuple[Any, ...], Mapping[str, int]]) -> Dict[str, Any]:
    if len(groups) < 3 or len(grouped) < 2:
        return {
            "test": "cochrans_q",
            "matched_units": len(grouped),
            "statistic": None,
            "p_value": None,
            "significant": False,
            "note": "Insufficient matched units or groups for Cochran's Q.",
        }

    matrix = [[value_map[group] for group in groups] for value_map in grouped.values()]
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(row[idx] for row in matrix) for idx in range(len(groups))]
    total = sum(col_sums)
    denominator = len(groups) * total - sum(value * value for value in row_sums)
    if denominator <= 0:
        return {
            "test": "cochrans_q",
            "matched_units": len(grouped),
            "statistic": None,
            "p_value": None,
            "significant": False,
            "note": "Degenerate Cochran's Q denominator.",
        }
    statistic = ((len(groups) - 1) * (len(groups) * sum(value * value for value in col_sums) - total * total)) / denominator
    p_value = float(1.0 - chi2.cdf(statistic, len(groups) - 1))
    return {
        "test": "cochrans_q",
        "matched_units": len(grouped),
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


def _group_comparison_stats(rows: List[Dict[str, Any]], group_field: str) -> Dict[str, Any]:
    groups = sorted({str(row[group_field]) for row in rows}, key=_natural_sort_key)
    if len(groups) < 2:
        return {
            "alpha": ALPHA,
            "group_field": group_field,
            "groups": groups,
            "matched": group_field in MATCHED_GROUP_FIELDS,
            "omnibus": None,
            "pairwise": [],
        }

    pairwise_results: List[Dict[str, Any]] = []
    matched = group_field in MATCHED_GROUP_FIELDS
    if matched:
        groups, grouped = _matched_values(rows, group_field)
        if len(groups) < 2 or not grouped:
            return {
                "alpha": ALPHA,
                "group_field": group_field,
                "groups": groups,
                "matched": True,
                "omnibus": None,
                "pairwise": [],
            }

        if len(groups) == 2:
            values = [(value_map[groups[0]], value_map[groups[1]]) for value_map in grouped.values()]
            omnibus = dict(_pairwise_exact_mcnemar(groups[0], groups[1], values))
            omnibus["test"] = "exact_mcnemar"
            omnibus["significant"] = omnibus["p_value"] < ALPHA
        else:
            omnibus = _cochrans_q(groups, grouped)

        raw_pairwise_p: List[float] = []
        for left, right in combinations(groups, 2):
            values = [(value_map[left], value_map[right]) for value_map in grouped.values()]
            result = _pairwise_exact_mcnemar(left, right, values)
            raw_pairwise_p.append(result["p_value"])
            pairwise_results.append(result)
    else:
        contingency: List[List[int]] = []
        counts_by_group: Dict[str, Tuple[int, int]] = {}
        for group in groups:
            subset = [row for row in rows if str(row[group_field]) == group]
            correct = sum(_safe_int(row["correct"]) for row in subset)
            total = len(subset)
            counts_by_group[group] = (correct, total - correct)
            contingency.append([correct, total - correct])

        total_correct = sum(correct for correct, _wrong in counts_by_group.values())
        total_wrong = sum(wrong for _correct, wrong in counts_by_group.values())
        if total_correct == 0 or total_wrong == 0:
            omnibus = {
                "test": "degenerate_accuracy_distribution",
                "statistic": None,
                "p_value": 1.0,
                "significant": False,
                "matched_units": None,
                "note": "All observations are correct or all are wrong, so no omnibus accuracy test is informative.",
            }
            raw_pairwise_p = []
            for left, right in combinations(groups, 2):
                left_correct, left_wrong = counts_by_group[left]
                right_correct, right_wrong = counts_by_group[right]
                pairwise_results.append(
                    {
                        "group_a": left,
                        "group_b": right,
                        "odds_ratio": None,
                        "p_value": 1.0,
                        "a_correct": left_correct,
                        "a_wrong": left_wrong,
                        "b_correct": right_correct,
                        "b_wrong": right_wrong,
                        "note": "Degenerate table: every observation has the same outcome.",
                    }
                )
                raw_pairwise_p.append(1.0)
            adjusted = _holm_adjust(raw_pairwise_p)
            for row, adjusted_p in zip(pairwise_results, adjusted):
                row["p_value_holm"] = adjusted_p
                row["significant_holm"] = False
            return {
                "alpha": ALPHA,
                "group_field": group_field,
                "groups": groups,
                "matched": matched,
                "omnibus": omnibus,
                "pairwise": pairwise_results,
            }

        if len(groups) == 2:
            statistic, p_value = fisher_exact(contingency)
            omnibus = {
                "test": "fisher_exact",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": float(p_value) < ALPHA,
                "matched_units": None,
            }
        else:
            try:
                statistic, p_value, degrees_of_freedom, _expected = chi2_contingency(contingency)
                omnibus = {
                    "test": "chi_square_independence",
                    "statistic": float(statistic),
                    "degrees_of_freedom": int(degrees_of_freedom),
                    "p_value": float(p_value),
                    "significant": float(p_value) < ALPHA,
                    "matched_units": None,
                }
            except ValueError as exc:
                omnibus = {
                    "test": "chi_square_independence",
                    "statistic": None,
                    "p_value": None,
                    "significant": False,
                    "matched_units": None,
                    "note": f"Chi-square test was not computable: {exc}",
                }

        raw_pairwise_p = []
        for left, right in combinations(groups, 2):
            left_correct, left_wrong = counts_by_group[left]
            right_correct, right_wrong = counts_by_group[right]
            statistic, p_value = fisher_exact(
                [[left_correct, left_wrong], [right_correct, right_wrong]]
            )
            raw_pairwise_p.append(float(p_value))
            pairwise_results.append(
                {
                    "group_a": left,
                    "group_b": right,
                    "odds_ratio": float(statistic),
                    "p_value": float(p_value),
                    "a_correct": left_correct,
                    "a_wrong": left_wrong,
                    "b_correct": right_correct,
                    "b_wrong": right_wrong,
                }
            )

    adjusted = _holm_adjust(raw_pairwise_p)
    for row, adjusted_p in zip(pairwise_results, adjusted):
        row["p_value_holm"] = adjusted_p
        row["significant_holm"] = adjusted_p < ALPHA

    return {
        "alpha": ALPHA,
        "group_field": group_field,
        "groups": groups,
        "matched": matched,
        "omnibus": omnibus,
        "pairwise": pairwise_results,
    }


def _build_plot_rows(summary_rows: List[Dict[str, Any]], x_field: str, series_field: str | None) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in summary_rows:
        output.append(
            {
                "x": row[x_field],
                "series": row[series_field] if series_field else "accuracy",
                "correct": _safe_int(row["correct"]),
                "total": _safe_int(row["total"]),
                "accuracy": _safe_float(row["accuracy"]),
                "standard_error": _safe_float(row["standard_error"]),
                "ci95_low": _safe_float(row["ci95_low"]),
                "ci95_high": _safe_float(row["ci95_high"]),
            }
        )
    return output


def _plot_grouped_bar(
    *,
    title: str,
    path: Path,
    x_labels: List[str],
    series_values: Dict[str, List[float]],
    series_errors: Dict[str, List[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(x_labels) * 1.5), 5))
    series_names = list(series_values.keys())
    width = 0.8 / max(len(series_names), 1)
    positions = list(range(len(x_labels)))

    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2) * width
        plt.bar(
            [pos + offset for pos in positions],
            series_values[series_name],
            yerr=series_errors[series_name],
            capsize=4,
            width=width,
            label=series_name,
        )

    plt.xticks(positions, x_labels, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    if len(series_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_line(
    *,
    title: str,
    path: Path,
    x_labels: List[str],
    values: List[float],
    errors: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(x_labels) * 1.2), 5))
    positions = list(range(len(x_labels)))
    plt.errorbar(positions, values, yerr=errors, marker="o", linestyle="-", capsize=4)
    plt.xticks(positions, x_labels)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_plot_bundle(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path.with_suffix(".json"), payload)


def _series_payload(
    *,
    title: str,
    x_field: str,
    series_field: str | None,
    filters: Mapping[str, Any],
    summary_rows: List[Dict[str, Any]],
    statistics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "title": title,
        "alpha": ALPHA,
        "x_field": x_field,
        "series_field": series_field,
        "filters": dict(filters),
        "rows": _build_plot_rows(summary_rows, x_field, series_field),
        "statistics": statistics,
    }


def _write_grouped_bar_outputs(
    *,
    title: str,
    path: Path,
    summary_rows: List[Dict[str, Any]],
    x_field: str,
    series_field: str | None,
    filters: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    x_labels = sorted({str(row[x_field]) for row in summary_rows}, key=_natural_sort_key)
    series_names = (
        sorted({str(row[series_field]) for row in summary_rows}, key=_natural_sort_key)
        if series_field
        else ["accuracy"]
    )
    series_values: Dict[str, List[float]] = {name: [] for name in series_names}
    series_errors: Dict[str, List[float]] = {name: [] for name in series_names}

    for series_name in series_names:
        for x_label in x_labels:
            matched = next(
                (
                    row
                    for row in summary_rows
                    if str(row[x_field]) == x_label
                    and (not series_field or str(row[series_field]) == series_name)
                ),
                None,
            )
            series_values[series_name].append(_safe_float(matched["accuracy"]) if matched else 0.0)
            series_errors[series_name].append(_safe_float(matched["standard_error"]) if matched else 0.0)

    _plot_grouped_bar(
        title=title,
        path=path,
        x_labels=x_labels,
        series_values=series_values,
        series_errors=series_errors,
    )
    _write_plot_bundle(
        path,
        _series_payload(
            title=title,
            x_field=x_field,
            series_field=series_field,
            filters=filters,
            summary_rows=summary_rows,
            statistics=statistics,
        ),
    )


def _write_line_outputs(
    *,
    title: str,
    path: Path,
    summary_rows: List[Dict[str, Any]],
    x_field: str,
    filters: Mapping[str, Any],
    statistics: Mapping[str, Any],
) -> None:
    ordered = sorted(summary_rows, key=lambda row: _natural_sort_key(row[x_field]))
    x_labels = [str(row[x_field]) for row in ordered]
    values = [_safe_float(row["accuracy"]) for row in ordered]
    errors = [_safe_float(row["standard_error"]) for row in ordered]
    _plot_line(title=title, path=path, x_labels=x_labels, values=values, errors=errors)
    _write_plot_bundle(
        path,
        _series_payload(
            title=title,
            x_field=x_field,
            series_field=None,
            filters=filters,
            summary_rows=ordered,
            statistics=statistics,
        ),
    )


def _generate_trend_plots(trial_rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    n_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    k_groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        n_groups[(row["model"], row["dataset"], row["prompt_type"], row["config_k"], row["config_q"])].append(row)
        k_groups[(row["model"], row["dataset"], row["prompt_type"], row["config_n"], row["config_q"])].append(row)

    for (model, dataset, prompt_type, config_k, config_q), rows in n_groups.items():
        if len({row["config_n"] for row in rows}) < 2:
            continue
        summary_rows = _aggregate_accuracy(rows, ["config_n"])
        path = plots_dir / (
            f"accuracy_vs_n__model_{_slugify(model)}__dataset_{_slugify(dataset)}__"
            f"prompt_{_slugify(prompt_type)}__k_{config_k}__q_{config_q}.png"
        )
        _write_line_outputs(
            title=f"Accuracy vs n | {model} | {dataset} | {prompt_type} | k={config_k}, q={config_q}",
            path=path,
            summary_rows=summary_rows,
            x_field="config_n",
            filters={"model": model, "dataset": dataset, "prompt_type": prompt_type, "config_k": config_k, "config_q": config_q},
            statistics=_group_comparison_stats(rows, "config_n"),
        )

    for (model, dataset, prompt_type, config_n, config_q), rows in k_groups.items():
        if len({row["config_k"] for row in rows}) < 2:
            continue
        summary_rows = _aggregate_accuracy(rows, ["config_k"])
        path = plots_dir / (
            f"accuracy_vs_k__model_{_slugify(model)}__dataset_{_slugify(dataset)}__"
            f"prompt_{_slugify(prompt_type)}__n_{config_n}__q_{config_q}.png"
        )
        _write_line_outputs(
            title=f"Accuracy vs k | {model} | {dataset} | {prompt_type} | n={config_n}, q={config_q}",
            path=path,
            summary_rows=summary_rows,
            x_field="config_k",
            filters={"model": model, "dataset": dataset, "prompt_type": prompt_type, "config_n": config_n, "config_q": config_q},
            statistics=_group_comparison_stats(rows, "config_k"),
        )


def _write_cross_model_plots(trial_rows: List[Dict[str, Any]], plots_dir: Path) -> List[str]:
    model_names = sorted({row["model"] for row in trial_rows}, key=_natural_sort_key)
    if len(model_names) < 2:
        return []

    generated: List[str] = []
    cross_specs = [
        ("prompt_type", "cross_model_accuracy_by_prompt.png", "Cross-Model Accuracy by Prompt Type"),
        ("dataset", "cross_model_accuracy_by_dataset.png", "Cross-Model Accuracy by Dataset"),
        ("config_n", "cross_model_accuracy_by_n.png", "Cross-Model Accuracy by n"),
    ]
    if len({row["config_k"] for row in trial_rows}) > 1:
        cross_specs.append(("config_k", "cross_model_accuracy_by_k.png", "Cross-Model Accuracy by k"))

    for x_field, filename, title in cross_specs:
        summary_rows = _aggregate_accuracy(trial_rows, [x_field, "model"])
        per_x_tests = []
        for x_value in sorted({row[x_field] for row in trial_rows}, key=_natural_sort_key):
            subset = [row for row in trial_rows if row[x_field] == x_value]
            per_x_tests.append({"x_value": x_value, **_group_comparison_stats(subset, "model")})
        _write_grouped_bar_outputs(
            title=title,
            path=plots_dir / filename,
            summary_rows=summary_rows,
            x_field=x_field,
            series_field="model",
            filters={},
            statistics={"per_x_level": per_x_tests},
        )
        generated.append(filename)

    overall_summary = _aggregate_accuracy(trial_rows, ["model"])
    _write_grouped_bar_outputs(
        title="Overall Accuracy by Model",
        path=plots_dir / "cross_model_overall_accuracy.png",
        summary_rows=overall_summary,
        x_field="model",
        series_field=None,
        filters={},
        statistics=_group_comparison_stats(trial_rows, "model"),
    )
    generated.append("cross_model_overall_accuracy.png")
    return generated


def analyze_run_directory(run_dir: Path) -> Dict[str, Path]:
    trial_results_path = run_dir / "trial_results.csv"
    run_accuracy_path = run_dir / "run_accuracy_long.csv"
    if not trial_results_path.exists():
        raise FileNotFoundError(f"Missing trial results file: {trial_results_path}")
    if not run_accuracy_path.exists():
        raise FileNotFoundError(f"Missing run accuracy file: {run_accuracy_path}")

    trial_rows = _read_csv(trial_results_path)
    analysis_dir = run_dir / "analysis"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"

    overall_prompt = _aggregate_accuracy(trial_rows, ["prompt_type"])
    dataset_prompt = _aggregate_accuracy(trial_rows, ["dataset", "prompt_type"])
    config_prompt = _aggregate_accuracy(trial_rows, ["config_n", "config_k", "config_q", "prompt_type"])

    _write_csv(
        tables_dir / "overall_accuracy_by_prompt.csv",
        ["prompt_type", "correct", "total", "accuracy", "standard_error", "ci95_low", "ci95_high"],
        overall_prompt,
    )
    _write_csv(
        tables_dir / "accuracy_by_dataset_and_prompt.csv",
        ["dataset", "prompt_type", "correct", "total", "accuracy", "standard_error", "ci95_low", "ci95_high"],
        dataset_prompt,
    )
    _write_csv(
        tables_dir / "accuracy_by_config_and_prompt.csv",
        ["config_n", "config_k", "config_q", "prompt_type", "correct", "total", "accuracy", "standard_error", "ci95_low", "ci95_high"],
        config_prompt,
    )

    _write_grouped_bar_outputs(
        title="Overall Accuracy by Prompt Type",
        path=plots_dir / "overall_accuracy_by_prompt.png",
        summary_rows=overall_prompt,
        x_field="prompt_type",
        series_field=None,
        filters={},
        statistics=_group_comparison_stats(trial_rows, "prompt_type"),
    )

    dataset_prompt_tests = []
    for dataset_name in sorted({row["dataset"] for row in trial_rows}, key=_natural_sort_key):
        subset = [row for row in trial_rows if row["dataset"] == dataset_name]
        dataset_prompt_tests.append({"dataset": dataset_name, **_group_comparison_stats(subset, "prompt_type")})
    _write_grouped_bar_outputs(
        title="Accuracy by Dataset and Prompt Type",
        path=plots_dir / "accuracy_by_dataset_and_prompt.png",
        summary_rows=dataset_prompt,
        x_field="dataset",
        series_field="prompt_type",
        filters={},
        statistics={"per_dataset": dataset_prompt_tests},
    )

    config_prompt_tests = []
    for config in sorted(
        {(row["config_n"], row["config_k"], row["config_q"]) for row in trial_rows},
        key=lambda item: tuple(_natural_sort_key(value) for value in item),
    ):
        subset = [
            row for row in trial_rows
            if row["config_n"] == config[0] and row["config_k"] == config[1] and row["config_q"] == config[2]
        ]
        config_prompt_tests.append(
            {
                "config_n": config[0],
                "config_k": config[1],
                "config_q": config[2],
                **_group_comparison_stats(subset, "prompt_type"),
            }
        )
    for row in config_prompt:
        row["config_label"] = f"N={row['config_n']},K={row['config_k']},Q={row['config_q']}"
    _write_grouped_bar_outputs(
        title="Accuracy by Few-Shot Configuration",
        path=plots_dir / "accuracy_by_config_and_prompt.png",
        summary_rows=config_prompt,
        x_field="config_label",
        series_field="prompt_type",
        filters={},
        statistics={"per_config": config_prompt_tests},
    )

    _generate_trend_plots(trial_rows, plots_dir)
    generated_cross_model = _write_cross_model_plots(trial_rows, plots_dir)

    report_lines = [
        "# OpenRouter Experiment Report",
        "",
        f"- Trials file: `{trial_results_path.name}`",
        f"- Run summaries: `{run_accuracy_path.name}`",
        f"- Models analyzed: {', '.join(sorted({row['model'] for row in trial_rows}, key=_natural_sort_key))}",
        "",
        "## Tables",
        "- `tables/overall_accuracy_by_prompt.csv`",
        "- `tables/accuracy_by_dataset_and_prompt.csv`",
        "- `tables/accuracy_by_config_and_prompt.csv`",
        "",
        "## Plots",
        "- `plots/overall_accuracy_by_prompt.png` + companion JSON",
        "- `plots/accuracy_by_dataset_and_prompt.png` + companion JSON",
        "- `plots/accuracy_by_config_and_prompt.png` + companion JSON",
        "- `plots/accuracy_vs_n__*.png` + companion JSON when multiple n values exist",
        "- `plots/accuracy_vs_k__*.png` + companion JSON when multiple k values exist",
    ]
    if generated_cross_model:
        report_lines.extend(
            [
                "",
                "## Cross-Model Plots",
                *[f"- `plots/{name}` + companion JSON" for name in generated_cross_model],
            ]
        )

    report_path = analysis_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "analysis_dir": analysis_dir,
        "report_path": report_path,
    }
