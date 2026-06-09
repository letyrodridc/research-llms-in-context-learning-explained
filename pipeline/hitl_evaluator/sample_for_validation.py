"""
Generates the 192-item stratified sample for HITL validation and produces
one annotator CSV per researcher (carmen, leticia, nico).

Usage (from repo root with research-explain conda env active):
    python pipeline/hitl_evaluator/sample_for_validation.py
"""

import json
import sys
import os
import pandas as pd
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_DIR   = os.path.join(REPO_ROOT, "pipeline", "openrouter_runs",
                         "full_experiment_single__20260427_213833")
TRIALS_CSV = os.path.join(RUN_DIR, "trial_results.csv")
JUDGE_CSV  = os.path.join(RUN_DIR, "judge_outputs", "openai-gpt-5-mini",
                          "judge_results.csv")
OUT_DIR    = os.path.dirname(__file__)

SEED = 42

# ── condition / model name mappings ───────────────────────────────────────────
CONDITION_MAP = {
    "nle":                 "E2",
    "features":            "E3",
    "rulebased":           "E4",
    "axioms_ontology_v2":  "E5",
}

MODEL_SHORT = {
    "google/gemini-2.5-flash":          "Gemini-2.5-Flash",
    "google/gemma-4-26b-a4b-it":        "Gemma-4-26B",
    "qwen/qwen3-vl-8b-instruct":        "Qwen3-VL-8B",
    "meta-llama/llama-4-scout":         "Llama-4-Scout",
}

# ── annotator block assignment ────────────────────────────────────────────────
# item_id is 1-indexed; blocks are 1-64, 65-128, 129-192
# A = Carmen + Leticia, B = Leticia + Nico, C = Carmen + Nico
ANNOTATOR_BLOCKS = {
    "carmen":  list(range(1, 65))   + list(range(129, 193)),   # A + C
    "leticia": list(range(1, 65))   + list(range(65, 129)),    # A + B
    "nico":    list(range(65, 129)) + list(range(129, 193)),   # B + C
}

# ── human annotation columns (empty in output) ───────────────────────────────
HUMAN_COLS = [
    "human_TG", "human_HF", "human_CC", "human_CP", "human_Cn",
    "human_S", "human_LD", "human_IF", "human_LC", "annotator_notes",
]


def load_and_merge() -> pd.DataFrame:
    """Loads trials + judge results, filters to E2-E5, inner-joins."""
    print("Loading trial_results.csv …")
    trials = pd.read_csv(TRIALS_CSV)
    print(f"  {len(trials):,} rows total")

    print("Loading judge_results.csv …")
    judge = pd.read_csv(JUDGE_CSV)
    print(f"  {len(judge):,} rows")

    # Filter to explanation conditions only (E2-E5)
    trials = trials[trials["prompt_type"].isin(CONDITION_MAP)].copy()
    print(f"  {len(trials):,} trials after filtering to E2-E5")

    # Join key
    join_key_trials = ["model", "dataset", "prompt_type",
                       "config_n", "config_k", "run_id", "query_index_within_episode"]
    join_key_judge  = ["source_model", "dataset", "prompt_type",
                       "config_n", "config_k", "run_id", "query_index_within_episode"]

    merged = trials.merge(
        judge,
        left_on=join_key_trials,
        right_on=join_key_judge,
        how="inner",
        suffixes=("", "_judge"),
    )
    print(f"  {len(merged):,} rows after inner join with judge")

    # Deduplicate judge-side duplicate columns (keep originals from trials)
    dup_cols = [c for c in merged.columns if c.endswith("_judge")]
    merged = merged.drop(columns=dup_cols)

    # Add derived columns
    merged["condition"]   = merged["prompt_type"].map(CONDITION_MAP)
    merged["model_short"] = merged["model"].map(MODEL_SHORT)
    merged["stratum"]     = (merged["condition"] + "_"
                             + merged["model_short"] + "_"
                             + merged["dataset"])
    return merged


def stratified_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    For each of 64 strata selects 3 items:
      - If ≥1 incorrect: 1 random incorrect + 2 random correct
      - If 0 incorrect:  3 random correct (exception documented)
    """
    rng = np.random.default_rng(seed)
    strata = sorted(df["stratum"].unique())
    expected_strata = len(CONDITION_MAP) * len(MODEL_SHORT) * 4  # 4 datasets
    assert len(strata) == expected_strata, (
        f"Expected {expected_strata} strata, got {len(strata)}"
    )

    selected_rows = []
    exceptions = []

    for stratum in strata:
        group = df[df["stratum"] == stratum]
        incorrect = group[group["correct"] == 0]
        correct   = group[group["correct"] == 1]

        if len(incorrect) >= 1:
            chosen_incorrect = incorrect.sample(n=1, random_state=int(rng.integers(1e9)))
            remaining_correct = correct.drop(chosen_incorrect.index, errors="ignore")
            if len(remaining_correct) >= 2:
                chosen_correct = remaining_correct.sample(n=2, random_state=int(rng.integers(1e9)))
            else:
                # Very unlikely: fewer than 2 correct after picking 1 incorrect
                chosen_correct = remaining_correct
            rows = pd.concat([chosen_incorrect, chosen_correct])
        else:
            # 0 incorrect in this stratum — fill all 3 with correct
            rows = correct.sample(n=min(3, len(correct)), random_state=int(rng.integers(1e9)))
            exceptions.append({
                "stratum": stratum,
                "n_incorrect": len(incorrect),
                "n_correct": len(correct),
                "n_selected": len(rows),
            })

        selected_rows.append(rows)

    if exceptions:
        print(f"\n⚠  EXCEPTIONS — strata with 0 incorrects ({len(exceptions)}):")
        for exc in exceptions:
            print(f"   {exc['stratum']}: {exc['n_incorrect']} incorrect, "
                  f"{exc['n_correct']} correct → selected {exc['n_selected']} correct")
    else:
        print("\n✓ All strata had ≥1 incorrect trial")

    sample = pd.concat(selected_rows, ignore_index=True)
    return sample, exceptions


def verify(sample: pd.DataFrame, annotator_dfs: dict) -> bool:
    """Runs all integrity checks. Returns True if all pass."""
    ok = True

    # 1. No duplicate item_ids
    dups = sample["item_id"].duplicated().sum()
    if dups:
        print(f"✗ FAIL: {dups} duplicate item_ids in sample")
        ok = False
    else:
        print("✓ No duplicate item_ids")

    # 2. Exactly 192 items
    if len(sample) != 192:
        print(f"✗ FAIL: sample has {len(sample)} items (expected 192)")
        ok = False
    else:
        print(f"✓ Sample has 192 items")

    # 3. Each item in exactly 2 annotator CSVs
    item_counts: dict[int, int] = {}
    for df in annotator_dfs.values():
        for iid in df["item_id"]:
            item_counts[iid] = item_counts.get(iid, 0) + 1
    bad = {iid: cnt for iid, cnt in item_counts.items() if cnt != 2}
    if bad:
        print(f"✗ FAIL: {len(bad)} items not in exactly 2 annotator CSVs")
        ok = False
    else:
        print("✓ Each item appears in exactly 2 annotator CSVs")

    # 4. Each annotator CSV has exactly 128 items
    for name, df in annotator_dfs.items():
        if len(df) != 128:
            print(f"✗ FAIL: {name} CSV has {len(df)} items (expected 128)")
            ok = False
        else:
            print(f"✓ {name}: 128 items")

    # 5. Stratum distribution (3 per stratum in sample)
    stratum_counts = sample["stratum"].value_counts()
    off = stratum_counts[stratum_counts != 3]
    if len(off):
        print(f"⚠  {len(off)} strata with ≠3 items (expected for 0-incorrect strata):")
        print(off)
    else:
        print("✓ All strata have exactly 3 items")

    return ok


def main():
    # ── load & merge ──────────────────────────────────────────────────────────
    df = load_and_merge()

    # ── stratified sample ─────────────────────────────────────────────────────
    print("\nRunning stratified sample (seed=42) …")
    sample, exceptions = stratified_sample(df, seed=SEED)

    # ── assign item_id (1-indexed, sorted by stratum for reproducibility) ─────
    sample = sample.sort_values("stratum").reset_index(drop=True)
    sample.insert(0, "item_id", range(1, len(sample) + 1))
    sample["selection_seed"] = SEED

    # ── build annotator CSVs ──────────────────────────────────────────────────
    annotator_dfs = {}
    for annotator, item_ids in ANNOTATOR_BLOCKS.items():
        adf = sample[sample["item_id"].isin(item_ids)].copy()
        adf = adf.sort_values("item_id").reset_index(drop=True)
        adf["annotator"] = annotator
        for col in HUMAN_COLS:
            adf[col] = ""
        annotator_dfs[annotator] = adf

    # ── verification ──────────────────────────────────────────────────────────
    print("\n── Verification ─────────────────────────────────────────────────────")
    all_ok = verify(sample, annotator_dfs)
    if not all_ok:
        print("\n✗ Verification failed — aborting without writing files")
        sys.exit(1)

    # ── audit report ──────────────────────────────────────────────────────────
    print("\n── Audit Report ─────────────────────────────────────────────────────")
    print(f"Total trials in experiment (E2-E5):  {len(df):,}")
    print(f"Unique strata:                        {df['stratum'].nunique()}")
    print(f"Items selected:                       {len(sample)}")
    print(f"Incorrect items in sample:            {(sample['correct'] == 0).sum()}")
    print(f"Correct items in sample:              {(sample['correct'] == 1).sum()}")
    print(f"Strata with exceptions (0 incorrect): {len(exceptions)}")
    print("\nItems per condition:")
    print(sample["condition"].value_counts().to_string())
    print("\nItems per model_short:")
    print(sample["model_short"].value_counts().to_string())
    print("\nItems per dataset:")
    print(sample["dataset"].value_counts().to_string())

    # ── save files ────────────────────────────────────────────────────────────
    sample_path = os.path.join(OUT_DIR, "human_validation_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"\n✓ Saved: {sample_path}")

    for annotator, adf in annotator_dfs.items():
        path = os.path.join(OUT_DIR, f"annotations_{annotator}.csv")
        adf.to_csv(path, index=False)
        print(f"✓ Saved: {path}")

    print("\nDone. Next step: git add + git push these CSVs so annotators can pull them.")


if __name__ == "__main__":
    main()
