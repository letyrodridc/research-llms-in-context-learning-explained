"""
HITL Annotation App — Flask server (local, 127.0.0.1).

Usage:
    conda activate /path/to/research-explain
    cd pipeline/hitl_evaluator
    python annotation_app/app.py --annotator carmen --data-dir ../../data

Then open http://127.0.0.1:8766 in your browser.
"""

import argparse
import io
import json
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# ── resolve repo root so we can import setup_utils ───────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
HITL_DIR  = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(HITL_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from pipeline.utils.setup_utils import load_datasets  # noqa: E402

app = Flask(__name__)

# ── globals set at startup ────────────────────────────────────────────────────
ANNOTATOR    = None
ANNOTATOR_DF = None          # rows from annotations_{annotator}.csv
DATASETS     = None          # dict from load_datasets()
PROGRESS     = None          # {annotations: {str(item_id): {...}}}
PROGRESS_FILE = None
CSV_PATH     = None

PORT_DEFAULTS = {"carmen": 8766, "leticia": 8767, "nico": 8768}


# ── helpers ───────────────────────────────────────────────────────────────────

def _pil_to_jpeg_bytes(pil_img) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _get_row(item_idx: int) -> pd.Series:
    """item_idx is 0-based position in ANNOTATOR_DF."""
    return ANNOTATOR_DF.iloc[item_idx]


def _dataset_image(dataset_name: str, dataset_index: int):
    """Load a PIL image from the torchvision dataset by index."""
    ds = DATASETS.get(dataset_name)
    if ds is None:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    img, _label = ds[dataset_index]
    # img is a PIL Image (transform in setup_utils keeps PIL via ToTensor? — check)
    # setup_utils applies a resize transform; img may be a Tensor if ToTensor was applied.
    # We need PIL. Try to convert.
    import torch
    from PIL import Image as PILImage
    if isinstance(img, PILImage.Image):
        return img
    if isinstance(img, torch.Tensor):
        import torchvision.transforms.functional as TF
        return TF.to_pil_image(img)
    raise TypeError(f"Unexpected image type: {type(img)}")


def _class_name(dataset_name: str, label) -> str:
    classes = DATASETS.get(f"{dataset_name}_classes", [])
    try:
        idx = int(label)
        return classes[idx] if idx < len(classes) else str(label)
    except (TypeError, ValueError):
        return str(label)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html", annotator=ANNOTATOR)


@app.get("/api/item/<int:item_idx>")
def get_item(item_idx: int):
    """Return metadata for item at 0-based position item_idx."""
    if item_idx < 0 or item_idx >= len(ANNOTATOR_DF):
        return jsonify({"error": "index out of range"}), 404

    row = _get_row(item_idx)
    support_indices = json.loads(row["support_indices"]) if isinstance(row["support_indices"], str) else list(row["support_indices"])

    item_id = int(row["item_id"])
    existing = PROGRESS["annotations"].get(str(item_id), {})

    explanation = (
        row.get("classifier_raw_response_text") or
        row.get("raw_response_text") or ""
    )

    return jsonify({
        "item_idx":          item_idx,
        "item_id":           item_id,
        "total":             len(ANNOTATOR_DF),
        "dataset":           row["dataset"],
        "condition":         row["condition"],
        "prompt_type":       row["prompt_type"],
        "model_short":       row["model_short"],
        "config_n":          int(row["config_n"]),
        "config_k":          int(row["config_k"]),
        "expected_label":    _class_name(row["dataset"], row["expected_label"]),
        "predicted_label":   _class_name(row["dataset"], row["predicted_label"]),
        "correct":           bool(int(row["correct"])),
        "stratum":           row["stratum"],
        "explanation":       str(explanation),
        "n_support":         len(support_indices),
        "existing_annotation": existing,
        # Judge scores (for reference display)
        "judge_scores": {
            "TG": row.get("textual_groundedness"),
            "HF": row.get("hallucination_free"),
            "CC": row.get("concept_counting"),
            "CP": row.get("comprehensibility"),
            "Cn": row.get("conciseness"),
            "S":  row.get("specificity"),
            "LD": row.get("discriminativeness"),
            "IF": row.get("instruction_following"),
            "LC": row.get("logical_coherence"),
        },
    })


@app.get("/api/items_index")
def items_index():
    """Lightweight list of all items for the navigation dropdown."""
    rows = []
    for idx, (_, row) in enumerate(ANNOTATOR_DF.iterrows()):
        item_id = int(row["item_id"])
        rows.append({
            "idx":        idx,
            "item_id":    item_id,
            "condition":  row["condition"],
            "prompt_type": row["prompt_type"],
            "dataset":    row["dataset"],
            "correct":    bool(int(row["correct"])),
            "annotated":  str(item_id) in PROGRESS["annotations"],
        })
    return jsonify(rows)


@app.get("/api/image/query/<int:item_idx>")
def query_image(item_idx: int):
    row = _get_row(item_idx)
    img = _dataset_image(row["dataset"], int(row["query_dataset_index"]))
    return send_file(io.BytesIO(_pil_to_jpeg_bytes(img)),
                     mimetype="image/jpeg")


@app.get("/api/image/support/<int:item_idx>/<int:k>")
def support_image(item_idx: int, k: int):
    row = _get_row(item_idx)
    support_indices = json.loads(row["support_indices"]) if isinstance(row["support_indices"], str) else list(row["support_indices"])
    if k >= len(support_indices):
        return jsonify({"error": "support index out of range"}), 404
    img = _dataset_image(row["dataset"], int(support_indices[k]))
    return send_file(io.BytesIO(_pil_to_jpeg_bytes(img)),
                     mimetype="image/jpeg")


@app.get("/api/progress")
def get_progress():
    annotated = list(PROGRESS["annotations"].keys())
    return jsonify({
        "total":           len(ANNOTATOR_DF),
        "annotated_count": len(annotated),
        "annotated_ids":   annotated,
        "annotations":     PROGRESS["annotations"],
    })


@app.post("/api/save")
def save_annotation():
    """Save annotation immediately to disk before responding."""
    payload = request.get_json(force=True)
    item_id = str(payload.get("item_id"))
    if not item_id:
        return jsonify({"error": "missing item_id"}), 400

    PROGRESS["annotations"][item_id] = payload

    # Write to disk BEFORE returning — critical for data safety
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(PROGRESS, f, indent=2, ensure_ascii=False)

    return jsonify({"status": "ok", "saved_item_id": item_id})


@app.get("/api/export")
def export_csv():
    """Generate final CSV with annotations merged into the base annotator CSV."""
    df = ANNOTATOR_DF.copy()

    human_cols = [
        "human_TG", "human_HF", "human_CC", "human_CP", "human_Cn",
        "human_S", "human_LD", "human_IF", "human_LC", "annotator_notes",
    ]

    for col in human_cols:
        df[col] = ""

    for item_id_str, ann in PROGRESS["annotations"].items():
        mask = df["item_id"] == int(item_id_str)
        for col in human_cols:
            if col in ann:
                df.loc[mask, col] = ann[col]

    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    filename = f"annotations_{ANNOTATOR}_final.csv"
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=filename)


# ── startup ───────────────────────────────────────────────────────────────────

def init_app(annotator: str, data_dir: str):
    global ANNOTATOR, ANNOTATOR_DF, DATASETS, PROGRESS, PROGRESS_FILE, CSV_PATH

    ANNOTATOR = annotator

    csv_path = os.path.join(HITL_DIR, f"annotations_{annotator}.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run sample_for_validation.py first to generate annotator CSVs.")
        sys.exit(1)

    CSV_PATH     = csv_path
    ANNOTATOR_DF = pd.read_csv(csv_path)
    print(f"Loaded {len(ANNOTATOR_DF)} items for annotator '{annotator}'")

    PROGRESS_FILE = os.path.join(HITL_DIR, f"progress_{annotator}.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            PROGRESS = json.load(f)
        done = len(PROGRESS.get("annotations", {}))
        print(f"Resumed progress: {done}/{len(ANNOTATOR_DF)} items already annotated")
    else:
        PROGRESS = {"annotator": annotator, "annotations": {}}
        print("No saved progress found — starting fresh")

    print(f"\nLoading datasets from {data_dir} …")
    print("(This may take a minute on first run; subsequent runs load from cache)\n")
    DATASETS = load_datasets(data_dir=data_dir)
    print("Datasets ready.\n")


def main():
    parser = argparse.ArgumentParser(description="HITL Annotation App")
    parser.add_argument("--annotator", required=True,
                        choices=["carmen", "leticia", "nico"],
                        help="Which annotator's CSV to load")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="Path to dataset root (default: <repo>/data)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port (default: 8766/8767/8768 per annotator)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    port = args.port or PORT_DEFAULTS.get(args.annotator, 8766)
    init_app(args.annotator, args.data_dir)

    print(f"Starting annotation app for '{args.annotator}'")
    print(f"Open http://{args.host}:{port} in your browser\n")
    app.run(host=args.host, port=port, debug=False)


if __name__ == "__main__":
    main()
