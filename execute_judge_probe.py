"""
Judge probe: runs the LLM judge on a representative sample
(1 trial per dataset × source-model × condition) from an experiment run,
reports actual token usage and cost, then extrapolates to the full run.

Usage:
    python execute_judge_probe.py \\
        --run-dir pipeline/openrouter_runs/<run_dir> \\
        --judge-model openai/gpt-5-mini

Output: console + a text report file in the current directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Published OpenRouter prices (USD per token).  Only used as fallback when
# the generation-cost API call fails.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "openai/gpt-5-mini":                (0.25e-6,  2.0e-6),
    "openai/gpt-5":                     (10.0e-6, 30.0e-6),
    "anthropic/claude-haiku-4-5":       ( 0.8e-6,  4.0e-6),
    "anthropic/claude-sonnet-4-6":      ( 3.0e-6, 15.0e-6),
    "google/gemini-2.5-flash":          (0.075e-6, 0.3e-6),
    "google/gemini-2.5-flash-lite":     (0.04e-6,  0.15e-6),
    "meta-llama/llama-4-scout":         (0.11e-6,  0.34e-6),
}

SEPARATOR = "─" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _select_representative_trials(
    trial_rows: List[Dict[str, str]],
    judgeable_types: Tuple[str, ...],
) -> List[Dict[str, str]]:
    """Pick the first valid trial for each (dataset, model, prompt_type) cell."""
    seen: set = set()
    selected: List[Dict[str, str]] = []

    # Sort deterministically by (dataset, model, prompt_type, run_id, query_index)
    def sort_key(r: Dict[str, str]) -> Tuple:
        return (
            r.get("dataset", ""),
            r.get("model", ""),
            r.get("prompt_type", ""),
            int(r.get("run_id", 0)),
            int(r.get("query_index_within_episode", 0)),
        )

    for row in sorted(trial_rows, key=sort_key):
        if row.get("prompt_type") not in judgeable_types:
            continue
        if row.get("error", "").strip():
            continue
        cell = (row.get("dataset"), row.get("model"), row.get("prompt_type"))
        if cell in seen:
            continue
        seen.add(cell)
        selected.append(row)

    return selected


def _fetch_generation_cost(api_key: str, generation_id: str) -> Optional[float]:
    """Query OpenRouter /api/v1/generation?id=X for actual cost in USD."""
    try:
        resp = requests.get(
            f"{OPENROUTER_BASE_URL}/generation",
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            cost = resp.json().get("data", {}).get("total_cost")
            if cost is not None:
                return float(cost)
    except Exception:
        pass
    return None


def _estimate_cost_from_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    in_price, out_price = pricing
    return prompt_tokens * in_price + completion_tokens * out_price


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def _fmt_cost(c: float) -> str:
    return f"${c:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the judge on a representative probe sample and report "
            "cost + extrapolation to the full run."
        )
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        nargs="+",
        required=True,
        help="One or more experiment run directories containing trial_results.csv.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model to use (overrides OPENROUTER_JUDGE_MODEL env var).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a .env file (default: repo-root/.env).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path for the text report (default: probe_report_<timestamp>.txt).",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Skip the OpenRouter model metadata check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    env_path = Path(args.env_file).resolve() if args.env_file else repo_root / ".env"
    _load_dotenv(env_path)

    # --- Pipeline imports (after sys.path and env are set up) ---
    from pipeline.experiments.config import (
        build_openrouter_settings,
        JUDGEABLE_PROMPT_TYPES,
    )
    from pipeline.utils.client import OpenRouterClient, model_supports_images
    from pipeline.evaluation.judge_prompts import build_judge_prompt_specs
    from pipeline.evaluation.reconstruction import reconstruct_classifier_messages
    from pipeline.evaluation.run_openrouter_judge import (
        build_judge_user_content,
        extract_scores_from_judge_response,
        flatten_system_prompt_into_first_user_message,
        is_developer_instruction_error,
    )
    from pipeline.utils.setup_utils import load_datasets, set_seed

    settings = build_openrouter_settings(
        env_path,
        cli_model=args.judge_model,
        env_model_key="OPENROUTER_JUDGE_MODEL",
        app_name_keys=("OPENROUTER_JUDGE_APP_NAME", "OPENROUTER_APP_NAME"),
        default_app_name="research-llms-icl-openrouter-judge-probe",
        timeout_keys=("OPENROUTER_JUDGE_TIMEOUT_SECONDS", "OPENROUTER_TIMEOUT_SECONDS"),
        retry_keys=("OPENROUTER_JUDGE_MAX_RETRIES", "OPENROUTER_MAX_RETRIES"),
    )
    client = OpenRouterClient(settings)

    if not args.skip_model_validation:
        model_info = client.fetch_model_metadata()
        if model_supports_images(model_info) is False:
            raise ValueError(
                f"Selected judge model does not advertise image support: {settings.model}"
            )
        if model_info is None:
            print(f"[!] Could not find metadata for {settings.model} in OpenRouter /models.")

    # --- Load all trial CSVs ---
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    all_trial_rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        csv_path = run_dir / "trial_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"trial_results.csv not found in {run_dir}")
        rows = _read_csv(csv_path)
        for row in rows:
            row["_source_run_dir"] = str(run_dir)
        all_trial_rows.extend(rows)

    total_judgeable = sum(
        1 for r in all_trial_rows
        if r.get("prompt_type") in JUDGEABLE_PROMPT_TYPES and not r.get("error", "").strip()
    )

    selected = _select_representative_trials(all_trial_rows, JUDGEABLE_PROMPT_TYPES)

    if not selected:
        raise ValueError("No valid judgeable trials found in the provided run directories.")

    set_seed(42)
    print("Loading datasets...", flush=True)
    datasets_dict = load_datasets(data_dir=str(repo_root / "data"))
    judge_prompt_specs = build_judge_prompt_specs(explain_scores=False)

    # --- Run probe ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_file) if args.output_file else Path(f"probe_report_{timestamp}.txt")

    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 72)
    emit("JUDGE PROBE REPORT")
    emit("=" * 72)
    emit(f"Generated      : {datetime.now().astimezone().isoformat()}")
    emit(f"Judge model    : {settings.model}")
    emit(f"Reasoning effort: medium  (hardcoded in run_openrouter_judge.py)")
    emit(f"Max tokens     : {judge_prompt_specs[list(judge_prompt_specs.keys())[0]].max_tokens:,}")
    emit(f"Run dir(s)     : {', '.join(str(d) for d in run_dirs)}")
    emit()
    emit(SEPARATOR)
    emit("SAMPLE SELECTION")
    emit(SEPARATOR)
    emit(f"  Total judgeable trials in full dataset : {total_judgeable:,}")
    emit(f"  Probe size                             : {len(selected)}")
    emit(f"  Selection strategy                     : 1 per (dataset × model × condition)")
    emit()

    # --- Judge loop ---
    emit(SEPARATOR)
    emit("TRIAL LOG")
    emit(SEPARATOR)

    probe_results: List[Dict[str, Any]] = []

    for idx, row in enumerate(selected, start=1):
        dataset_name = row["dataset"]
        model_name   = row["model"]
        prompt_type  = row["prompt_type"]
        label = f"{dataset_name:10s} / {model_name.split('/')[-1]:28s} / {prompt_type}"
        prefix = f"  [{idx:3d}/{len(selected)}]  {label}"

        source_run_dir = Path(row["_source_run_dir"])
        dataset = datasets_dict[dataset_name]
        class_names = datasets_dict.get(f"{dataset_name}_classes")
        spec = judge_prompt_specs[prompt_type]

        print(f"{prefix}  ...", flush=True)
        try:
            trial = reconstruct_classifier_messages(row, dataset, class_names)
            user_content = build_judge_user_content(row, trial.classifier_messages)
            messages = [
                {"role": "system", "content": spec.system_prompt},
                {"role": "user",   "content": user_content},
            ]
            print(f"{prefix}  calling judge...", flush=True)
            try:
                response = client.create_chat_completion(
                    messages=messages,
                    max_tokens=spec.max_tokens,
                    temperature=0.0,
                    generation_params={"reasoning_effort": "medium"},
                )
            except Exception as exc:
                if is_developer_instruction_error(exc):
                    messages = flatten_system_prompt_into_first_user_message(messages)
                    response = client.create_chat_completion(
                        messages=messages,
                        max_tokens=spec.max_tokens,
                        temperature=0.0,
                        generation_params={"reasoning_effort": "medium"},
                    )
                else:
                    raise

            scores, parse_error = extract_scores_from_judge_response(response.text)
            prompt_tokens     = int(response.usage.get("prompt_tokens", 0))
            completion_tokens = int(response.usage.get("completion_tokens", 0))
            total_tokens      = int(response.usage.get("total_tokens", 0)) or (prompt_tokens + completion_tokens)
            finish_reason     = response.finish_reason or ""
            response_id       = response.request_id or ""

            status_tag = "SUCCESS" if not parse_error else f"PARSE_WARN ({parse_error[:40]})"
            emit(
                f"{prefix}  {status_tag}  "
                f"tokens: {_fmt_tokens(prompt_tokens)}+{_fmt_tokens(completion_tokens)}"
                f"={_fmt_tokens(total_tokens)}"
                + (f"  finish={finish_reason}" if finish_reason != "stop" else "")
            )

            probe_results.append({
                "cell": (dataset_name, model_name, prompt_type),
                "success": True,
                "parse_error": parse_error,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_id": response_id,
                "actual_cost": None,
            })

        except Exception as exc:
            emit(f"{prefix}  FAILED  {type(exc).__name__}: {str(exc)[:80]}")
            probe_results.append({
                "cell": (dataset_name, model_name, prompt_type),
                "success": False,
                "parse_error": str(exc),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "response_id": "",
                "actual_cost": None,
            })

    # --- Fetch actual costs from OpenRouter generation API ---
    emit()
    emit(SEPARATOR)
    emit("FETCHING ACTUAL COSTS  (OpenRouter /api/v1/generation)")
    emit(SEPARATOR)
    emit("  Waiting 3 s for OpenRouter to settle...")
    time.sleep(3)

    cost_source = "estimated from token counts"
    any_actual_cost = False
    for result in probe_results:
        if not result["success"]:
            continue
        rid = result["response_id"]
        if rid:
            actual = _fetch_generation_cost(settings.api_key, rid)
            if actual is not None:
                result["actual_cost"] = actual
                any_actual_cost = True
                continue
        # Fallback: estimate from token counts
        est = _estimate_cost_from_tokens(
            settings.model,
            result["prompt_tokens"],
            result["completion_tokens"],
        )
        result["actual_cost"] = est

    if any_actual_cost:
        cost_source = "OpenRouter generation API (actual)"
        emit(f"  Cost source : {cost_source}")
    else:
        emit(f"  Cost source : {cost_source}")
        emit(f"  (Generation API unavailable — falling back to token × published price)")

    # --- Summary ---
    successful = [r for r in probe_results if r["success"]]
    failed     = [r for r in probe_results if not r["success"]]
    n_success  = len(successful)
    n_failed   = len(failed)

    costed     = [r for r in successful if r["actual_cost"] is not None]
    costs      = [r["actual_cost"] for r in costed]

    emit()
    emit("=" * 72)
    emit("SUMMARY")
    emit("=" * 72)
    emit()
    emit(f"  Probe trials      : {len(probe_results)}")
    emit(f"  Successful        : {n_success}  ({100*n_success/max(len(probe_results),1):.1f}%)")
    emit(f"  Failed            : {n_failed}")
    emit()

    if successful:
        all_prompt     = [r["prompt_tokens"]     for r in successful]
        all_completion = [r["completion_tokens"] for r in successful]
        all_total      = [r["total_tokens"]      for r in successful]
        emit("  Token usage (successful trials)")
        emit(f"    Input  mean/trial : {_fmt_tokens(int(mean(all_prompt)))}")
        emit(f"    Output mean/trial : {_fmt_tokens(int(mean(all_completion)))}")
        emit(f"    Total  mean/trial : {_fmt_tokens(int(mean(all_total)))}")
        emit(f"    Probe total       : {_fmt_tokens(sum(all_total))} tokens")
        emit()

    if costs:
        mean_cost  = mean(costs)
        total_cost = sum(costs)
        emit(f"  Cost ({n_success} successful trials)")
        emit(f"    Source            : {cost_source}")
        emit(f"    Mean per trial    : {_fmt_cost(mean_cost)}")
        emit(f"    Probe total       : {_fmt_cost(total_cost)}")
        emit()
        emit(SEPARATOR)
        emit("  EXTRAPOLATION TO FULL RUN")
        emit(SEPARATOR)
        emit(f"    Total judgeable trials : {total_judgeable:,}")
        emit(f"    Mean cost per trial    : {_fmt_cost(mean_cost)}")
        extrapolated = mean_cost * total_judgeable
        emit(f"    Estimated total cost   : {_fmt_cost(extrapolated)}")
        emit(f"    Conservative (+20%)    : {_fmt_cost(extrapolated * 1.2)}")
        emit(f"    Optimistic   (−20%)    : {_fmt_cost(extrapolated * 0.8)}")
        emit()
        emit("    NOTE: estimate assumes same judge model, reasoning_effort=medium,")
        emit(f"    max_tokens={judge_prompt_specs[list(judge_prompt_specs.keys())[0]].max_tokens:,}.")
        emit("    Actual cost varies with explanation length per trial.")
    else:
        emit("  [!] No cost data available — could not estimate total cost.")

    if failed:
        emit()
        emit(SEPARATOR)
        emit("  FAILED TRIALS")
        emit(SEPARATOR)
        for r in failed:
            ds, mdl, pt = r["cell"]
            emit(f"    {ds} / {mdl} / {pt}  —  {r['parse_error'][:80]}")

    emit()
    emit(SEPARATOR)
    emit(f"  Report written to : {output_file.resolve()}")
    emit("=" * 72)

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
