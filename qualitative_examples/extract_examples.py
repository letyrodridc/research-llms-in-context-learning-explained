"""
Extract representative qualitative examples for TFM analysis chapter.
"""

import pandas as pd
import json
import os
import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
RUN_DIR = Path("/mnt/homeGPU/cquiles/ICL/ICL/pipeline/openrouter_runs/full_experiment_single__20260427_213833")
JUDGE_CSV = RUN_DIR / "judge_outputs/openai-gpt-5-mini/judge_results.csv"
MODELS_DIR = RUN_DIR / "models"
OUT_DIR = Path("/mnt/homeGPU/cquiles/ICL/ICL/qualitative_examples")
DASH_DIR = OUT_DIR / "dashboard_examples_run"

MODEL_SLUGS = {
    "google/gemini-2.5-flash":      "google-gemini-2.5-flash",
    "google/gemma-4-26b-a4b-it":    "google-gemma-4-26b-a4b-it",
    "qwen/qwen3-vl-8b-instruct":    "qwen-qwen3-vl-8b-instruct",
    "meta-llama/llama-4-scout":     "meta-llama-llama-4-scout",
}

CONDITION_MAP = {
    "nle":               "E2_NLE",
    "features":          "E3_FEATURES",
    "rulebased":         "E4_RULEBASED",
    "axioms_ontology_v2": "E5_AXIOMS",
    "classification":    "E1_CLASSIFICATION",
}

MERGE_KEYS = ["dataset", "prompt_type", "config_n", "config_k", "config_q",
              "run_id", "query_index_within_episode"]

SCORE_COLS = ["textual_groundedness", "hallucination_free", "concept_counting",
              "comprehensibility", "conciseness", "specificity", "discriminativeness",
              "instruction_following", "logical_coherence", "overall_score"]

# ─── 1. Load & merge ──────────────────────────────────────────────────────────
print("Loading trial results...")
trials_frames = []
for model_id, slug in MODEL_SLUGS.items():
    df = pd.read_csv(MODELS_DIR / slug / "trial_results.csv")
    df["model_slug"] = slug
    trials_frames.append(df)

trials = pd.concat(trials_frames, ignore_index=True)
print(f"  Total trials: {len(trials)}")

print("Loading judge results...")
judge = pd.read_csv(JUDGE_CSV)
# Keep only rows without parse errors and with all scores valid
judge_clean = judge[judge["judge_parse_error"].isna()].copy()
judge_clean = judge_clean.dropna(subset=SCORE_COLS)
print(f"  Total judge rows (clean): {len(judge_clean)} / {len(judge)}")

# Merge: judge has source_model, trials has model
judge_clean = judge_clean.rename(columns={"source_model": "model"})
merged = trials.merge(
    judge_clean[MERGE_KEYS + ["model"] + SCORE_COLS + ["judge_parse_error", "judge_raw_response_text"]],
    on=MERGE_KEYS + ["model"],
    how="inner"
)
print(f"  Merged rows: {len(merged)}")

# ─── Helper ───────────────────────────────────────────────────────────────────
def build_entry(case_id, row):
    cond = CONDITION_MAP.get(row["prompt_type"], row["prompt_type"])
    scores = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in SCORE_COLS}
    return {
        "case_id": case_id,
        "dataset": row["dataset"],
        "model": row["model"],
        "model_slug": row["model_slug"],
        "condition": cond,
        "prompt_type": row["prompt_type"],
        "config_n": int(row["config_n"]),
        "config_k": int(row["config_k"]),
        "run_id": int(row["run_id"]),
        "query_index_within_episode": int(row["query_index_within_episode"]),
        "expected_label": str(row["expected_label"]),
        "predicted_label": str(row["predicted_label"]),
        "correct": int(row["correct"]),
        "raw_response_text": str(row["raw_response_text"]),
        "episode_filepath": str(row["episode_filepath"]),
        "query_dataset_index": int(row["query_dataset_index"]) if pd.notna(row["query_dataset_index"]) else None,
        "conversation_log_path": str(row.get("conversation_log_path", "")),
        "artifact_dir": str(row.get("artifact_dir", "")),
        "judge_scores": scores,
    }


def pick_best(df, sort_cols, ascending=False):
    if df.empty:
        return None
    return df.sort_values(sort_cols, ascending=ascending).iloc[0]


def pick_worst(df, sort_cols):
    return pick_best(df, sort_cols, ascending=True)


# ─── 2. Extract each case ─────────────────────────────────────────────────────
results = {}
used_trial_indices = {}  # model_slug -> set of original trial df indices

def register(entry, orig_idx, model_slug):
    if model_slug not in used_trial_indices:
        used_trial_indices[model_slug] = set()
    used_trial_indices[model_slug].add(orig_idx)

# Helper to get the original trial index
def get_trial_idx(row, model_slug):
    mask = (
        (trials["model_slug"] == model_slug) &
        (trials["dataset"] == row["dataset"]) &
        (trials["prompt_type"] == row["prompt_type"]) &
        (trials["config_n"] == row["config_n"]) &
        (trials["config_k"] == row["config_k"]) &
        (trials["run_id"] == row["run_id"]) &
        (trials["query_index_within_episode"] == row["query_index_within_episode"])
    )
    idxs = trials[mask].index.tolist()
    return idxs[0] if idxs else None


# ── E2 NLE (one per dataset) ──
nle = merged[merged["prompt_type"] == "nle"]
for dataset, case_id in [("flowers","EXITO_E2_FLOWERS"), ("pets","EXITO_E2_PETS"),
                           ("cifar10","EXITO_E2_CIFAR10"), ("dtd","EXITO_E2_DTD")]:
    pool = nle[(nle["dataset"] == dataset) & (nle["correct"] == 1) &
               (nle["logical_coherence"] >= 4) & (nle["textual_groundedness"] >= 4)].copy()
    pool["_sort"] = pool["logical_coherence"] + pool["textual_groundedness"] + pool["overall_score"]
    row = pick_best(pool, "_sort")
    if row is not None:
        entry = build_entry(case_id, row)
        results[case_id] = entry
        idx = get_trial_idx(row, row["model_slug"])
        if idx is not None:
            register(entry, idx, row["model_slug"])
        print(f"  {case_id}: model={row['model']}, LD={row['logical_coherence']}, TG={row['textual_groundedness']}, overall={row['overall_score']}")
    else:
        print(f"  {case_id}: NOT FOUND")

# ── E3 Features ──
feat = merged[merged["prompt_type"] == "features"]
pool = feat[(feat["correct"] == 1) & (feat["logical_coherence"] >= 4) & (feat["instruction_following"] == 5)].copy()
row = pick_best(pool, "overall_score")
if row is not None:
    entry = build_entry("EXITO_E3_FEATURES", row)
    results["EXITO_E3_FEATURES"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  EXITO_E3_FEATURES: model={row['model']}, dataset={row['dataset']}, LD={row['logical_coherence']}, IF={row['instruction_following']}, overall={row['overall_score']}")
else:
    print("  EXITO_E3_FEATURES: NOT FOUND")

# ── E4 Rule-based (feature-value pairs) ──
rb = merged[merged["prompt_type"] == "rulebased"]
pool = rb[(rb["correct"] == 1) & (rb["logical_coherence"] >= 4) & (rb["specificity"] >= 4)].copy()
row = pick_best(pool, "overall_score")
if row is not None:
    entry = build_entry("EXITO_E4_RULEBASED", row)
    results["EXITO_E4_RULEBASED"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  EXITO_E4_RULEBASED: model={row['model']}, dataset={row['dataset']}, LD={row['logical_coherence']}, S={row['specificity']}, overall={row['overall_score']}")
else:
    print("  EXITO_E4_RULEBASED: NOT FOUND")

# ── E5 DL Axioms ──
ax = merged[merged["prompt_type"] == "axioms_ontology_v2"]
pool = ax[(ax["correct"] == 1) & (ax["model"] != "qwen/qwen3-vl-8b-instruct")].copy()
row = pick_best(pool, "overall_score")
if row is not None:
    entry = build_entry("EXITO_E5_AXIOMS", row)
    results["EXITO_E5_AXIOMS"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  EXITO_E5_AXIOMS: model={row['model']}, dataset={row['dataset']}, overall={row['overall_score']}")
else:
    print("  EXITO_E5_AXIOMS: NOT FOUND")

# ── Fallo: accuracy sin quality ──
expl = merged[merged["prompt_type"].isin(["nle", "features", "rulebased"])]
pool = expl[(expl["correct"] == 1) & (expl["logical_coherence"] <= 2) & (expl["textual_groundedness"] <= 2)].copy()
row = pick_worst(pool, "overall_score")
if row is not None:
    entry = build_entry("FALLO_ACCURACY_SIN_QUALITY", row)
    results["FALLO_ACCURACY_SIN_QUALITY"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  FALLO_ACCURACY_SIN_QUALITY: model={row['model']}, pt={row['prompt_type']}, LD={row['logical_coherence']}, TG={row['textual_groundedness']}, overall={row['overall_score']}")
else:
    print("  FALLO_ACCURACY_SIN_QUALITY: NOT FOUND")

# ── Fallo: quality sin accuracy ──
pool = expl[(expl["correct"] == 0) & (expl["logical_coherence"] >= 4)].copy()
row = pick_best(pool, "overall_score")
if row is not None:
    entry = build_entry("FALLO_QUALITY_SIN_ACCURACY", row)
    results["FALLO_QUALITY_SIN_ACCURACY"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  FALLO_QUALITY_SIN_ACCURACY: model={row['model']}, pt={row['prompt_type']}, dataset={row['dataset']}, LD={row['logical_coherence']}, overall={row['overall_score']}")
else:
    print("  FALLO_QUALITY_SIN_ACCURACY: NOT FOUND")

# ── Colapso Qwen E5 ──
pool_qwen = ax[ax["model"] == "qwen/qwen3-vl-8b-instruct"].copy()
if pool_qwen.empty:
    print("  COLAPSO_QWEN_E5: NOT FOUND (no qwen axioms rows)")
else:
    min_tg = pool_qwen["textual_groundedness"].min()
    pool_q = pool_qwen[pool_qwen["textual_groundedness"] == min_tg].copy()
    pool_q["_sort"] = pool_q["textual_groundedness"] + pool_q["logical_coherence"] + pool_q["overall_score"]
    row = pick_worst(pool_q, "_sort")
    if row is not None:
        entry = build_entry("COLAPSO_QWEN_E5", row)
        results["COLAPSO_QWEN_E5"] = entry
        idx = get_trial_idx(row, row["model_slug"])
        if idx is not None: register(entry, idx, row["model_slug"])
        print(f"  COLAPSO_QWEN_E5: TG={row['textual_groundedness']}, LD={row['logical_coherence']}, overall={row['overall_score']}, dataset={row['dataset']}")

# ── Excepción LLaMA DTD E5 ──
pool = ax[(ax["model"] == "meta-llama/llama-4-scout") & (ax["dataset"] == "dtd") & (ax["correct"] == 1)].copy()
row = pick_best(pool, "overall_score")
if row is not None:
    entry = build_entry("EXCEPCION_LLAMA_DTD_E5", row)
    results["EXCEPCION_LLAMA_DTD_E5"] = entry
    idx = get_trial_idx(row, row["model_slug"])
    if idx is not None: register(entry, idx, row["model_slug"])
    print(f"  EXCEPCION_LLAMA_DTD_E5: overall={row['overall_score']}, TG={row['textual_groundedness']}, LD={row['logical_coherence']}")
else:
    print("  EXCEPCION_LLAMA_DTD_E5: NOT FOUND")

# ── Comparativa multi-modelo ──
print("\nSearching for COMPARATIVA_MULTI_MODELO...")
all_4_models = set(MODEL_SLUGS.keys())
comp_pt = merged[merged["prompt_type"].isin(["features", "rulebased"])].copy()
group_keys = ["dataset", "prompt_type", "config_n", "config_k", "run_id", "query_index_within_episode"]
grouped = comp_pt.groupby(group_keys)

best_combo = None
best_score_var = -1
best_rows = None

for key, grp in grouped:
    if len(grp["model"].unique()) < 4:
        continue
    if not all_4_models.issubset(set(grp["model"].unique())):
        continue
    # Measure divergence by variance in overall_score + label disagreement
    score_var = grp["overall_score"].var()
    n_correct = grp["correct"].sum()
    label_disagreement = 1 if (n_correct > 0 and n_correct < 4) else 0
    combined = score_var + label_disagreement * 2
    if combined > best_score_var:
        best_score_var = combined
        best_combo = key
        best_rows = grp

if best_rows is not None:
    print(f"  Best combo: {best_combo}, score_var={best_score_var:.3f}")
    comp_label = {"google/gemini-2.5-flash": "COMPARATIVA_GEMINI",
                  "google/gemma-4-26b-a4b-it": "COMPARATIVA_GEMMA",
                  "qwen/qwen3-vl-8b-instruct": "COMPARATIVA_QWEN",
                  "meta-llama/llama-4-scout": "COMPARATIVA_LLAMA"}
    comp_entries = []
    for _, row in best_rows.iterrows():
        cid = comp_label.get(row["model"], f"COMPARATIVA_{row['model']}")
        entry = build_entry(cid, row)
        comp_entries.append(entry)
        idx = get_trial_idx(row, row["model_slug"])
        if idx is not None: register(entry, idx, row["model_slug"])
        print(f"    {cid}: correct={row['correct']}, overall={row['overall_score']}")
    results["COMPARATIVA_MULTI_MODELO"] = comp_entries
else:
    print("  COMPARATIVA_MULTI_MODELO: NOT FOUND")

# ─── 3. Save JSON ─────────────────────────────────────────────────────────────
out_json = OUT_DIR / "qualitative_examples.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved {out_json}")

# ─── 4. Build mini dashboard run-dir ─────────────────────────────────────────
print("\nBuilding dashboard mini run-dir...")
DASH_DIR.mkdir(parents=True, exist_ok=True)

# Collect all selected trial rows per model
selected_per_model = {slug: [] for slug in MODEL_SLUGS.values()}

for case_id, entry in results.items():
    if case_id == "COMPARATIVA_MULTI_MODELO":
        entries = entry  # list
    else:
        entries = [entry]
    for e in entries:
        slug = e["model_slug"]
        selected_per_model[slug].append(e)

# Write per-model trial_results.csv subsets
for slug, model_id in [(v,k) for k,v in MODEL_SLUGS.items()]:
    model_entries = selected_per_model[slug]
    if not model_entries:
        continue
    # Get original trial rows for these entries
    full_trial_df = pd.read_csv(MODELS_DIR / slug / "trial_results.csv")
    mask = pd.Series([False] * len(full_trial_df))
    for e in model_entries:
        row_mask = (
            (full_trial_df["dataset"] == e["dataset"]) &
            (full_trial_df["prompt_type"] == e["prompt_type"]) &
            (full_trial_df["config_n"] == e["config_n"]) &
            (full_trial_df["config_k"] == e["config_k"]) &
            (full_trial_df["run_id"] == e["run_id"]) &
            (full_trial_df["query_index_within_episode"] == e["query_index_within_episode"])
        )
        mask = mask | row_mask
    subset = full_trial_df[mask]
    model_out_dir = DASH_DIR / "models" / slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(model_out_dir / "trial_results.csv", index=False)
    print(f"  {slug}: {len(subset)} trial rows written")

# Write judge_results.csv subset
judge_out_dir = DASH_DIR / "judge_outputs" / "openai-gpt-5-mini"
judge_out_dir.mkdir(parents=True, exist_ok=True)

# Collect all (model, dataset, prompt_type, config_n, config_k, run_id, query_index) tuples
selected_keys = []
for case_id, entry in results.items():
    entries = entry if case_id == "COMPARATIVA_MULTI_MODELO" else [entry]
    for e in entries:
        selected_keys.append({
            "model": e["model"],
            "dataset": e["dataset"],
            "prompt_type": e["prompt_type"],
            "config_n": e["config_n"],
            "config_k": e["config_k"],
            "run_id": e["run_id"],
            "query_index_within_episode": e["query_index_within_episode"],
        })

judge_full = pd.read_csv(JUDGE_CSV)
judge_full_renamed = judge_full.rename(columns={"source_model": "model"})
mask = pd.Series([False] * len(judge_full_renamed))
for sk in selected_keys:
    row_mask = (
        (judge_full_renamed["model"] == sk["model"]) &
        (judge_full_renamed["dataset"] == sk["dataset"]) &
        (judge_full_renamed["prompt_type"] == sk["prompt_type"]) &
        (judge_full_renamed["config_n"] == sk["config_n"]) &
        (judge_full_renamed["config_k"] == sk["config_k"]) &
        (judge_full_renamed["run_id"] == sk["run_id"]) &
        (judge_full_renamed["query_index_within_episode"] == sk["query_index_within_episode"])
    )
    mask = mask | row_mask

judge_subset = judge_full[mask].copy()
# Fix source_run_dir to point to original
judge_subset["source_run_dir"] = str(RUN_DIR)
judge_out_dir.mkdir(parents=True, exist_ok=True)
judge_subset.to_csv(judge_out_dir / "judge_results.csv", index=False)
print(f"  Judge subset: {len(judge_subset)} rows written")

print(f"\nDashboard run-dir: {DASH_DIR}")
print(f"Launch command:")
print(f"  python pipeline/dashboard/run_results_dashboard.py --run-dir {DASH_DIR}")
print("\nDone.")
