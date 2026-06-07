from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import csv
import json
import re
import threading
import webbrowser

from ..utils.prompt_assets import repo_root
from ..experiments.prompts import _tensor_to_pil
from ..evaluation.reconstruction import reconstruct_classifier_messages


def _read_csv_safe(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_field(value: str, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _extract_xml_blocks(text: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    for match in re.finditer(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", text, flags=re.DOTALL):
        blocks.append({"tag": match.group(1), "content": match.group(2).strip()})
    return blocks


def _browser_messages(
    classifier_messages: List[Dict[str, Any]],
    image_refs: List[Any],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    image_cursor = 0
    browser_messages: List[Dict[str, Any]] = []
    for message in classifier_messages:
        content = message.get("content")
        if isinstance(content, str):
            browser_messages.append({"role": message.get("role"), "content": content})
            continue

        parts: List[Dict[str, Any]] = []
        for part in content or []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
                continue
            if part.get("type") == "image_url":
                ref = image_refs[image_cursor]
                parts.append(
                    {
                        "type": "image_ref",
                        "image_ref": {
                            "kind": ref.kind,
                            "dataset_index": ref.dataset_index,
                            "order_index": ref.order_index,
                            "dataset": dataset_name,
                            "url": f"/api/image?dataset={dataset_name}&index={ref.dataset_index}",
                        },
                    }
                )
                image_cursor += 1
                continue
            parts.append(dict(part))
        browser_messages.append({"role": message.get("role"), "content": parts})
    return browser_messages


class RunDataStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.repo_root = repo_root()
        self.trial_rows = _read_csv_safe(self.run_dir / "trial_results.csv")
        self.summary_rows = _read_csv_safe(self.run_dir / "experiment_summary.csv")
        self.run_rows = _read_csv_safe(self.run_dir / "run_accuracy_long.csv")
        
        # Fallback for local experiments which use a different summary filename
        if not self.summary_rows:
            self.summary_rows = _read_csv_safe(self.run_dir / "results_summary.csv")
            
        self.experiment_config = _read_json_if_exists(self.run_dir / "experiment_config.json")
        self.experiment_snapshot = _read_json_if_exists(self.run_dir / "experiment_config_snapshot.json")
        self.prompt_library = _read_json_if_exists(self.run_dir / "prompt_library_snapshot.json")
        self.run_manifest = _read_json_if_exists(self.run_dir / "run_manifest.json")
        self._datasets_cache: Optional[Dict[str, Any]] = None

        # Build trial lookup for cross-referencing from judge trials
        self._trial_lookup: Dict[str, Dict[str, str]] = {}
        for row in self.trial_rows:
            key = f"{row['dataset']}_{row['prompt_type']}_{row.get('model', '')}_{row['run_id']}_{row['query_index_within_episode']}"
            self._trial_lookup[key] = row

        # Load Judge Results if they exist (supports both legacy flat layout
        # `judge_outputs/judge_results_*.csv` and the unified per-judge-model
        # subdir layout `judge_outputs/<judge_model>/judge_results.csv`).
        self.judge_data: Dict[str, Dict[str, str]] = {}
        self.judge_rows: List[Dict[str, str]] = []
        judge_dir = self.run_dir / "judge_outputs"
        if judge_dir.exists():
            for pattern in ("judge_results_*.csv", "*/judge_results.csv"):
                for judge_file in sorted(judge_dir.glob(pattern)):
                    j_rows = _read_csv_safe(judge_file)
                    for jr in j_rows:
                        query_idx = jr.get("query_index_within_episode") or jr.get("query_index", "")
                        source_model = jr.get("source_model", "")
                        key = f"{jr['dataset']}_{jr['prompt_type']}_{source_model}_{jr['run_id']}_{query_idx}"
                        self.judge_data[key] = jr
                        jr["_judge_trial_id"] = str(len(self.judge_rows))
                        self.judge_rows.append(jr)

        for idx, row in enumerate(self.trial_rows):
            row["_trial_id"] = str(idx)
            # Link judge data to trial row
            model = row.get("model", "")
            key = f"{row['dataset']}_{row['prompt_type']}_{model}_{row['run_id']}_{row['query_index_within_episode']}"
            if key in self.judge_data:
                row["_judge_scores"] = self.judge_data[key]

    def _load_datasets(self) -> Dict[str, Any]:
        if self._datasets_cache is None:
            from ..utils.setup_utils import load_datasets

            self._datasets_cache = load_datasets(data_dir=str(self.repo_root / "data"))
        return self._datasets_cache

    def list_trials(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for row in self.trial_rows:
            class_id_map = _parse_json_field(row.get("class_id_map", ""), {})
            expected_id = row["expected_label"]
            predicted_id = row["predicted_label"]
            output.append(
                {
                    "trial_id": row["_trial_id"],
                    "dataset": row["dataset"],
                    "prompt_type": row["prompt_type"],
                    "model": row.get("model", ""),
                    "config_n": row.get("config_n") or row.get("n", "N/A"),
                    "config_k": row.get("config_k") or row.get("k", "N/A"),
                    "config_q": row.get("config_q") or row.get("q", "N/A"),
                    "run_id": row["run_id"],
                    "query_index_within_episode": row["query_index_within_episode"],
                    "expected_label": expected_id,
                    "predicted_label": predicted_id,
                    "expected_name": class_id_map.get(expected_id, expected_id),
                    "predicted_name": class_id_map.get(predicted_id, predicted_id),
                    "correct": row["correct"],
                    "parse_issue": row.get("parse_issue", ""),
                    "warning": row.get("warning", ""),
                    "error": row.get("error", ""),
                    "latency_seconds": row.get("latency_seconds", ""),
                    "usage_total_tokens": row.get("usage_total_tokens", ""),
                    "judge_scores": row.get("_judge_scores"),
                }
            )
        return output

    def get_trial(self, trial_id: str) -> Dict[str, Any]:
        row = self.trial_rows[int(trial_id)]
        datasets = self._load_datasets()
        dataset_name = row["dataset"]
        dataset = datasets[dataset_name]
        class_names = datasets.get(f"{dataset_name}_classes")
        reconstructed = reconstruct_classifier_messages(row, dataset, class_names)

        return {
            "trial_id": row["_trial_id"],
            "metadata": {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            },
            "judge_scores": row.get("_judge_scores"),
            "class_options": _parse_json_field(row.get("class_options", ""), []),
            "class_id_map": _parse_json_field(row.get("class_id_map", ""), {}),
            "image_refs": _parse_json_field(row.get("image_refs", ""), []),
            "conversation": _browser_messages(
                reconstructed.classifier_messages,
                reconstructed.image_refs,
                dataset_name,
            ),
            "raw_response": row.get("raw_response_text", ""),
            "parsed_response": _extract_xml_blocks(row.get("raw_response_text", "")),
        }

    def list_judge_trials(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for jr in self.judge_rows:
            query_idx = jr.get("query_index_within_episode", "")
            source_model = jr.get("source_model", "")
            trial_key = f"{jr.get('dataset', '')}_{jr.get('prompt_type', '')}_{source_model}_{jr.get('run_id', '')}_{query_idx}"
            source = self._trial_lookup.get(trial_key, {})
            output.append(
                {
                    "judge_trial_id": jr["_judge_trial_id"],
                    "judge_model": jr.get("judge_model", ""),
                    "judge_mode": jr.get("judge_mode", ""),
                    "source_model": source_model,
                    "dataset": jr.get("dataset", ""),
                    "prompt_type": jr.get("prompt_type", ""),
                    "config_n": jr.get("config_n", ""),
                    "config_k": jr.get("config_k", ""),
                    "config_q": jr.get("config_q", ""),
                    "run_id": jr.get("run_id", ""),
                    "query_index_within_episode": query_idx,
                    "overall_score": jr.get("overall_score", ""),
                    "judge_parse_error": jr.get("judge_parse_error", ""),
                    "source_correct": source.get("correct", ""),
                    "source_expected_label": source.get("expected_label", ""),
                    "source_predicted_label": source.get("predicted_label", ""),
                }
            )
        return output

    def get_judge_trial(self, judge_trial_id: str) -> Dict[str, Any]:
        jr = self.judge_rows[int(judge_trial_id)]
        dataset_name = jr.get("dataset", "")
        prompt_type = jr.get("prompt_type", "")
        run_id = jr.get("run_id", "")
        query_idx = jr.get("query_index_within_episode", "")

        source_model = jr.get("source_model", "")
        trial_key = f"{dataset_name}_{prompt_type}_{source_model}_{run_id}_{query_idx}"
        source = self._trial_lookup.get(trial_key, {})
        query_dataset_index = source.get("query_dataset_index", "")
        class_id_map = _parse_json_field(source.get("class_id_map", ""), {})

        raw_preview = jr.get("judge_message_preview", "")
        judge_messages_preview = _parse_json_field(raw_preview, [])

        # Build ordered image list matching what was actually sent to the judge.
        # num_support_images_shown=0 for query_only, >0 for query_and_support.
        support_indices_raw = _parse_json_field(source.get("support_indices", ""), [])
        num_support_shown = int(jr.get("num_support_images_shown", 0) or 0)
        pil_image_refs: List[tuple] = [
            ("support", int(idx)) for idx in support_indices_raw[:num_support_shown]
        ]
        if query_dataset_index:
            pil_image_refs.append(("query", int(query_dataset_index)))
        pil_cursor = 0

        # Replace image placeholders with real API URLs.
        # Handles both OpenRouter-style {"type": "image_url"} and local-judge
        # {"type": "image", "format": "pil"} placeholders.
        # Both advance the same pil_cursor so order is preserved.
        browser_judge_messages: List[Dict[str, Any]] = []
        for msg in judge_messages_preview:
            content = msg.get("content")
            if isinstance(content, str):
                browser_judge_messages.append({"role": msg.get("role"), "content": content})
                continue
            new_parts: List[Dict[str, Any]] = []
            for part in (content or []):
                if part.get("type") == "image_url" and pil_cursor < len(pil_image_refs):
                    kind, idx = pil_image_refs[pil_cursor]
                    pil_cursor += 1
                    new_parts.append(
                        {
                            "type": "image_ref",
                            "image_ref": {
                                "kind": kind,
                                "dataset_index": idx,
                                "dataset": dataset_name,
                                "url": f"/api/image?dataset={dataset_name}&index={idx}",
                            },
                        }
                    )
                elif part.get("type") == "image" and part.get("format") == "pil" and pil_cursor < len(pil_image_refs):
                    kind, idx = pil_image_refs[pil_cursor]
                    pil_cursor += 1
                    new_parts.append(
                        {
                            "type": "image_ref",
                            "image_ref": {
                                "kind": kind,
                                "dataset_index": idx,
                                "dataset": dataset_name,
                                "url": f"/api/image?dataset={dataset_name}&index={idx}",
                            },
                        }
                    )
                else:
                    new_parts.append(part)
            browser_judge_messages.append({"role": msg.get("role"), "content": new_parts})

        return {
            "judge_trial_id": judge_trial_id,
            "metadata": {k: v for k, v in jr.items() if not k.startswith("_")},
            "judge_messages": browser_judge_messages,
            "judge_raw_response": jr.get("judge_raw_response_text", ""),
            "parsed_judge_response": _extract_xml_blocks(jr.get("judge_raw_response_text", "")),
            "class_id_map": class_id_map,
            "source_trial": {
                "correct": source.get("correct", ""),
                "expected_label": source.get("expected_label", ""),
                "predicted_label": source.get("predicted_label", ""),
                "class_id_map": class_id_map,
            } if source else None,
        }

    def load_image_bytes(self, dataset_name: str, dataset_index: int) -> bytes:
        datasets = self._load_datasets()
        dataset = datasets[dataset_name]
        image_tensor, _label = dataset[dataset_index]
        image = _tensor_to_pil(image_tensor)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return buffer.getvalue()

    def summary_payload(self) -> Dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_name": self.run_dir.name,
            "run_manifest": self.run_manifest,
            "experiment_config": self.experiment_config,
            "experiment_snapshot": self.experiment_snapshot,
            "prompt_library_snapshot": self.prompt_library,
            "experiment_summary": self.summary_rows,
            "run_accuracy_long": self.run_rows,
            "trial_count": len(self.trial_rows),
        }


def _dashboard_html() -> str:
    return r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OpenRouter Run Dashboard</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f5f2ea; color: #1f2430; }

    /* Tab bar */
    .tab-bar { display: flex; gap: 0; background: #2c2a26; padding: 0 16px; }
    .tab-btn { padding: 12px 24px; color: #c8b89a; font-size: 14px; font-weight: 600; cursor: pointer; border: none; background: none; border-bottom: 3px solid transparent; transition: color 0.15s, border-color 0.15s; }
    .tab-btn:hover { color: #ffe8b5; }
    .tab-btn.active { color: #ffe8b5; border-bottom-color: #f5a623; }

    /* Tab panels */
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    .layout { display: grid; grid-template-columns: 440px 1fr; min-height: calc(100vh - 44px); }
    .sidebar { background: #fffaf1; border-right: 1px solid #d9d0bf; padding: 16px; overflow: auto; display: flex; flex-direction: column; gap: 10px; }
    .main { padding: 20px; overflow: auto; }
    h2, h3 { margin: 0 0 10px; }
    .card { background: white; border: 1px solid #d9d0bf; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

    details.section { background: white; border: 1px solid #d9d0bf; border-radius: 12px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); overflow: hidden; }
    details.section > summary { padding: 12px 14px; font-weight: 700; cursor: pointer; user-select: none; list-style: none; display: flex; align-items: center; gap: 8px; background: #f7f2e8; }
    details.section > summary::-webkit-details-marker { display: none; }
    details.section > summary::before { content: '▶'; font-size: 11px; transition: transform 0.15s; }
    details.section[open] > summary::before { transform: rotate(90deg); }
    details.section > .section-body { padding: 14px; }

    .trial-row { padding: 10px; border-top: 1px solid #ece4d7; cursor: pointer; }
    .trial-row:hover { background: #fff7e3; }
    .trial-row.active { background: #ffe8b5; }
    .ok { color: #0a7f3f; font-weight: 600; }
    .bad { color: #a12a2a; font-weight: 600; }
    .warn { color: #8b5e00; }
    pre { white-space: pre-wrap; word-break: break-word; background: #faf7f0; padding: 10px; border-radius: 8px; border: 1px solid #e5dccb; margin: 0; }
    .message { border: 1px solid #ddd1be; border-radius: 12px; margin-bottom: 12px; background: white; overflow: hidden; }
    .message-header { padding: 8px 12px; font-weight: 700; background: #f1eadc; text-transform: capitalize; }
    .message-body { padding: 12px; }
    .content-part { margin-bottom: 10px; }
    .img-wrap { display: inline-block; margin: 4px 8px 4px 0; vertical-align: top; }
    .img-label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; color: #5a4e3a; }
    .img-label.support { color: #1a6aa0; }
    .img-label.query { color: #7b3f00; }
    img.preview { max-width: 200px; max-height: 200px; border-radius: 10px; border: 2px solid #d8cdb7; display: block; }
    img.preview.query-img { border-color: #c47b2a; }
    img.preview.support-img { border-color: #4a90d9; }
    .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .filters label { font-size: 12px; font-weight: 600; color: #5a4e3a; }
    .filters select { width: 100%; padding: 6px 8px; border-radius: 8px; border: 1px solid #c9bea9; background: white; font-size: 13px; }
    .filter-full { grid-column: 1 / -1; }
    .trial-count { font-size: 12px; color: #7a6e5e; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ede4d6; font-size: 13px; }
    th { background: #f7f2e8; font-weight: 600; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #efe6d3; margin-right: 4px; font-size: 12px; }
    .run-meta { font-size: 13px; }
    .run-meta strong { font-size: 15px; display: block; margin-bottom: 4px; }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .stat { background: #fff7e3; border-radius: 10px; padding: 10px; font-size: 13px; }
    .explanation-box { background: #fff8e1; border: 2px solid #f9a825; border-radius: 8px; padding: 10px; }
  </style>
</head>
<body>
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('classification')">Classification Results</button>
    <button class="tab-btn" onclick="switchTab('judge')">Judge Evaluation</button>
  </div>

  <!-- ─── CLASSIFICATION TAB ─── -->
  <div id="tab-classification" class="tab-panel active">
    <div class="layout">
      <div class="sidebar">
        <div id="runMeta" class="card run-meta"></div>
        <details class="section" open>
          <summary>Trials</summary>
          <div class="section-body">
            <div class="filters">
              <div class="filter-full">
                <label for="modelFilter">Model</label>
                <select id="modelFilter"></select>
              </div>
              <div>
                <label for="datasetFilter">Dataset</label>
                <select id="datasetFilter"></select>
              </div>
              <div>
                <label for="promptFilter">Condition</label>
                <select id="promptFilter"></select>
              </div>
            </div>
            <div id="trialCount" class="trial-count"></div>
            <div id="trialList"></div>
          </div>
        </details>
      </div>
      <div class="main">
        <details class="section" open>
          <summary>Experiment Summary</summary>
          <div class="section-body" id="summaryBody"></div>
        </details>
        <details class="section">
          <summary>Config Snapshot</summary>
          <div class="section-body" id="configBody"></div>
        </details>
        <div id="trialDetail"></div>
      </div>
    </div>
  </div>

  <!-- ─── JUDGE TAB ─── -->
  <div id="tab-judge" class="tab-panel">
    <div class="layout">
      <div class="sidebar">
        <div id="judgeRunMeta" class="card run-meta"></div>
        <details class="section" open>
          <summary>Judge Trials</summary>
          <div class="section-body">
            <div class="filters">
              <div class="filter-full">
                <label for="judgeModelFilter">Judge Model</label>
                <select id="judgeModelFilter"></select>
              </div>
              <div class="filter-full">
                <label for="sourceModelFilter">Source Model</label>
                <select id="sourceModelFilter"></select>
              </div>
              <div class="filter-full">
                <label for="judgeModeFilter">Judge Mode</label>
                <select id="judgeModeFilter"></select>
              </div>
              <div>
                <label for="judgeDatasetFilter">Dataset</label>
                <select id="judgeDatasetFilter"></select>
              </div>
              <div>
                <label for="judgePromptFilter">Condition</label>
                <select id="judgePromptFilter"></select>
              </div>
            </div>
            <div id="judgeTrialCount" class="trial-count"></div>
            <div id="judgeTrialList"></div>
          </div>
        </details>
      </div>
      <div class="main">
        <div id="judgeTrialDetail"></div>
      </div>
    </div>
  </div>

  <script>
    const PROMPT_LABELS = {
      'classification':       'Classification without explanation',
      'nle':                  'Natural language explanation',
      'features':             'Features-based explanation',
      'rulebased':            'Feature-value based explanations',
      'axioms_ontology_v2':   'DL Axioms',
    };
    const JUDGE_DIMENSIONS = [
      ['textual_groundedness',  'Textual Groundedness'],
      ['hallucination_free',    'Hallucination free'],
      ['concept_counting',      'Concept counting'],
      ['comprehensibility',     'Comprehensibility'],
      ['conciseness',           'Conciseness'],
      ['specificity',           'Specificity'],
      ['discriminativeness',    'Discriminativeness'],
      ['instruction_following', 'Instruction following'],
      ['logical_coherence',     'Logical coherence'],
    ];
    const DATASET_LABELS = {
      'flowers': 'Flowers 102',
      'pets':    'Oxford Pets',
      'cifar10': 'CIFAR-10',
      'dtd':     'DTD',
    };
    function labelPrompt(v) { return PROMPT_LABELS[v] || v; }
    function labelDataset(v) { return DATASET_LABELS[v] || v; }
    function labelModel(v) { return v ? v.split('/').pop() : v; }

    // ── Tab switching ──────────────────────────────────────────────
    function switchTab(name) {
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');
      document.querySelector(`.tab-btn[onclick="switchTab('${name}')"]`).classList.add('active');
      if (name === 'judge' && allJudgeTrials.length === 0) loadJudgeTrials();
    }

    // ── Shared helpers ─────────────────────────────────────────────
    let allTrials = [];
    let activeTrialId = null;
    let allJudgeTrials = [];
    let activeJudgeTrialId = null;

    function escapeHtml(value) {
      return (value ?? '').toString()
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function renderMessages(messages) {
      return messages.map(message => {
        if (typeof message.content === 'string') {
          return `<div class="message">
            <div class="message-header">${escapeHtml(message.role)}</div>
            <div class="message-body"><pre>${escapeHtml(message.content)}</pre></div>
          </div>`;
        }
        const parts = (message.content || []).map(part => {
          if (part.type === 'text') {
            return `<div class="content-part"><pre>${escapeHtml(part.text)}</pre></div>`;
          }
          if (part.type === 'image_ref') {
            const kind = part.image_ref.kind || 'query';
            return `<div class="content-part img-wrap">
              <div class="img-label ${kind}">${escapeHtml(kind)}</div>
              <img class="preview ${kind}-img" src="${part.image_ref.url}" />
            </div>`;
          }
          return `<div class="content-part"><pre>${escapeHtml(JSON.stringify(part, null, 2))}</pre></div>`;
        }).join('');
        return `<div class="message">
          <div class="message-header">${escapeHtml(message.role)}</div>
          <div class="message-body">${parts}</div>
        </div>`;
      }).join('');
    }

    function renderScoresGrid(scores) {
      return JUDGE_DIMENSIONS.map(([key, label]) => {
        const val = scores[key];
        const display = val === undefined || val === '' ? '—' : `${escapeHtml(String(val))}/5`;
        return `<div class="stat"><strong>${escapeHtml(label)}</strong><br/>${display}</div>`;
      }).join('');
    }

    // ── Classification tab ─────────────────────────────────────────
    async function loadSummary() {
      const summary = await fetch('/api/summary').then(r => r.json());
      const metaHtml = `
        <strong>${escapeHtml(summary.run_name)}</strong>
        <div style="color:#7a6e5e;font-size:12px;">${escapeHtml(summary.run_dir)}</div>
        <div style="margin-top:8px;"><span class="pill">Trials: ${summary.trial_count}</span></div>
      `;
      document.getElementById('runMeta').innerHTML = metaHtml;
      document.getElementById('judgeRunMeta').innerHTML = metaHtml;

      const rows = (summary.experiment_summary || []).map(row => `
        <tr>
          <td>${escapeHtml(labelPrompt(row.prompt_type))}</td>
          <td>${escapeHtml(row.overall_accuracy)}</td>
          <td>${escapeHtml(row.total_trials)}</td>
          <td>${escapeHtml(row.total_errors)}</td>
        </tr>
      `).join('');
      document.getElementById('summaryBody').innerHTML = `
        <table>
          <thead><tr><th>Condition</th><th>Accuracy</th><th>Trials</th><th>Errors</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;
      document.getElementById('configBody').innerHTML = `
        <pre>${escapeHtml(JSON.stringify(summary.experiment_config || summary.experiment_snapshot || {}, null, 2))}</pre>
      `;
    }

    async function loadTrials() {
      allTrials = await fetch('/api/trials').then(r => r.json());

      const models   = ['all', ...new Set(allTrials.map(t => t.model).filter(Boolean))];
      const datasets = ['all', ...new Set(allTrials.map(t => t.dataset))];
      const prompts  = ['all', ...new Set(allTrials.map(t => t.prompt_type))];

      document.getElementById('modelFilter').innerHTML =
        models.map(v => `<option value="${v}">${v === 'all' ? 'All models' : escapeHtml(labelModel(v))}</option>`).join('');
      document.getElementById('datasetFilter').innerHTML =
        datasets.map(v => `<option value="${v}">${v === 'all' ? 'All datasets' : escapeHtml(labelDataset(v))}</option>`).join('');
      document.getElementById('promptFilter').innerHTML =
        prompts.map(v => `<option value="${v}">${v === 'all' ? 'All conditions' : escapeHtml(labelPrompt(v))}</option>`).join('');

      document.getElementById('modelFilter').addEventListener('change', renderTrialList);
      document.getElementById('datasetFilter').addEventListener('change', renderTrialList);
      document.getElementById('promptFilter').addEventListener('change', renderTrialList);
      renderTrialList();
    }

    function renderTrialList() {
      const modelF   = document.getElementById('modelFilter').value;
      const datasetF = document.getElementById('datasetFilter').value;
      const promptF  = document.getElementById('promptFilter').value;

      const filtered = allTrials.filter(t =>
        (modelF   === 'all' || t.model       === modelF)   &&
        (datasetF === 'all' || t.dataset     === datasetF) &&
        (promptF  === 'all' || t.prompt_type === promptF)
      );

      document.getElementById('trialCount').textContent = `${filtered.length} trial${filtered.length !== 1 ? 's' : ''}`;
      document.getElementById('trialList').innerHTML = filtered.map(t => `
        <div class="trial-row ${t.trial_id === activeTrialId ? 'active' : ''}" onclick="showTrial('${t.trial_id}')">
          <div style="display:flex; justify-content: space-between; align-items: center;">
            <strong>${escapeHtml(labelPrompt(t.prompt_type))}</strong>
            ${t.judge_scores ? `<span class="pill" style="background:#e3f2fd;">Judge: ${escapeHtml(t.judge_scores.overall_score || '?')}</span>` : ''}
          </div>
          <div style="font-size:12px;color:#5a4e3a;">${escapeHtml(labelDataset(t.dataset))} &nbsp;·&nbsp; ${escapeHtml(labelModel(t.model))}</div>
          <div style="font-size:12px;">run ${escapeHtml(t.run_id)} &nbsp;·&nbsp; expected: <em>${escapeHtml(t.expected_name)}</em> &nbsp;·&nbsp; predicted: <em>${escapeHtml(t.predicted_name)}</em></div>
          <div class="${t.correct === '1' ? 'ok' : 'bad'}">${t.correct === '1' ? '✓ correct' : '✗ incorrect'}</div>
          ${(t.parse_issue || t.warning || t.error) ? `<div class="warn">⚠ ${escapeHtml(t.parse_issue || t.warning || t.error)}</div>` : ''}
        </div>
      `).join('');
    }

    async function showTrial(trialId) {
      activeTrialId = trialId;
      renderTrialList();
      const detail = await fetch(`/api/trial?trial_id=${trialId}`).then(r => r.json());

      const meta = detail.metadata;
      const classIdMap = detail.class_id_map || {};
      const expectedId = meta.expected_label || '';
      const predictedId = meta.predicted_label || '';
      const expectedName = classIdMap[expectedId] || expectedId;
      const predictedName = classIdMap[predictedId] || predictedId;

      const messages = renderMessages(detail.conversation);

      const parsed = detail.parsed_response.length
        ? detail.parsed_response.map(b => `<div style="margin-bottom:10px;"><strong>&lt;${escapeHtml(b.tag)}&gt;</strong><pre style="margin-top:6px;">${escapeHtml(b.content)}</pre></div>`).join('')
        : '<div style="color:#7a6e5e;">No XML blocks parsed from the model response.</div>';

      const judgeHtml = detail.judge_scores ? (() => {
        const stats = renderScoresGrid(detail.judge_scores);
        const overall = detail.judge_scores.overall_score || '?';
        const critique = detail.judge_scores.judge_raw_response_text || detail.judge_scores.raw_judge_output || '';
        return `
        <details class="section" open>
          <summary>👨‍⚖️ Judge Evaluation &nbsp; <span class="pill" style="background:#e3f2fd;">Overall: ${escapeHtml(overall)}</span></summary>
          <div class="section-body">
            <div class="stat-grid" style="grid-template-columns: repeat(5, 1fr);">${stats}</div>
            <div style="margin-top:12px;">
              <strong>Full Critique:</strong>
              <pre style="font-size:0.9em; max-height: 240px; overflow-y: auto;">${escapeHtml(critique)}</pre>
            </div>
          </div>
        </details>`;
      })() : '';

      document.getElementById('trialDetail').innerHTML = `
        ${judgeHtml}
        <details class="section" open>
          <summary>Trial Metadata</summary>
          <div class="section-body">
            <table>
              <tr><th>Condition</th><td>${escapeHtml(labelPrompt(meta.prompt_type || ''))}</td></tr>
              <tr><th>Dataset</th><td>${escapeHtml(labelDataset(meta.dataset || ''))}</td></tr>
              <tr><th>Model</th><td>${escapeHtml(meta.model || '')}</td></tr>
              <tr><th>Config</th><td>${escapeHtml(meta.config_n)}‑way &nbsp;${escapeHtml(meta.config_k)}‑shot</td></tr>
              <tr><th>Run / Query</th><td>run ${escapeHtml(meta.run_id)} &nbsp;·&nbsp; query ${escapeHtml(meta.query_index_within_episode)}</td></tr>
              <tr><th>Expected</th><td><strong>${escapeHtml(expectedName)}</strong> <span style="color:#7a6e5e;font-size:12px;">(id=${escapeHtml(expectedId)})</span></td></tr>
              <tr><th>Predicted</th><td><strong class="${meta.correct === '1' ? 'ok' : 'bad'}">${escapeHtml(predictedName)}</strong> <span style="color:#7a6e5e;font-size:12px;">(id=${escapeHtml(predictedId)})</span></td></tr>
              <tr><th>Result</th><td class="${meta.correct === '1' ? 'ok' : 'bad'}">${meta.correct === '1' ? '✓ correct' : '✗ incorrect'}</td></tr>
              <tr><th>Class IDs</th><td><pre style="font-size:12px;">${escapeHtml(JSON.stringify(classIdMap, null, 2))}</pre></td></tr>
              ${meta.latency_seconds ? `<tr><th>Latency</th><td>${escapeHtml(meta.latency_seconds)} s</td></tr>` : ''}
              ${meta.usage_total_tokens ? `<tr><th>Tokens</th><td>${escapeHtml(meta.usage_total_tokens)}</td></tr>` : ''}
              ${meta.parse_issue ? `<tr><th>Parse issue</th><td class="warn">${escapeHtml(meta.parse_issue)}</td></tr>` : ''}
              ${meta.error ? `<tr><th>Error</th><td class="bad">${escapeHtml(meta.error)}</td></tr>` : ''}
            </table>
          </div>
        </details>
        <details class="section" open>
          <summary>Conversation</summary>
          <div class="section-body">${messages}</div>
        </details>
        <details class="section" open>
          <summary>Model Response</summary>
          <div class="section-body">${parsed}</div>
        </details>
        <details class="section">
          <summary>📝 Full Model Output (raw)</summary>
          <div class="section-body"><pre style="font-size: 0.95em; background: #fffde7; border: 1px solid #ffe082;">${escapeHtml(detail.raw_response || '')}</pre></div>
        </details>
      `;
    }

    // ── Judge tab ──────────────────────────────────────────────────
    async function loadJudgeTrials() {
      allJudgeTrials = await fetch('/api/judge-trials').then(r => r.json());
      if (allJudgeTrials.length === 0) {
        document.getElementById('judgeTrialList').innerHTML = '<div style="padding:16px;color:#7a6e5e;">No judge results found for this run.</div>';
        document.getElementById('judgeTrialCount').textContent = '0 trials';
        return;
      }

      const judgeModels  = ['all', ...new Set(allJudgeTrials.map(t => t.judge_model).filter(Boolean))];
      const srcModels    = ['all', ...new Set(allJudgeTrials.map(t => t.source_model).filter(Boolean))];
      const judgeModes   = ['all', ...new Set(allJudgeTrials.map(t => t.judge_mode).filter(Boolean))];
      const datasets     = ['all', ...new Set(allJudgeTrials.map(t => t.dataset))];
      const prompts      = ['all', ...new Set(allJudgeTrials.map(t => t.prompt_type))];

      const MODE_LABELS = { 'query_only': 'Query only', 'query_and_support': 'Query + support' };

      document.getElementById('judgeModelFilter').innerHTML =
        judgeModels.map(v => `<option value="${v}">${v === 'all' ? 'All judge models' : escapeHtml(labelModel(v))}</option>`).join('');
      document.getElementById('sourceModelFilter').innerHTML =
        srcModels.map(v => `<option value="${v}">${v === 'all' ? 'All source models' : escapeHtml(labelModel(v))}</option>`).join('');
      document.getElementById('judgeModeFilter').innerHTML =
        judgeModes.map(v => `<option value="${v}">${v === 'all' ? 'Both modes' : escapeHtml(MODE_LABELS[v] || v)}</option>`).join('');
      document.getElementById('judgeDatasetFilter').innerHTML =
        datasets.map(v => `<option value="${v}">${v === 'all' ? 'All datasets' : escapeHtml(labelDataset(v))}</option>`).join('');
      document.getElementById('judgePromptFilter').innerHTML =
        prompts.map(v => `<option value="${v}">${v === 'all' ? 'All conditions' : escapeHtml(labelPrompt(v))}</option>`).join('');

      document.getElementById('judgeModelFilter').addEventListener('change', renderJudgeList);
      document.getElementById('sourceModelFilter').addEventListener('change', renderJudgeList);
      document.getElementById('judgeModeFilter').addEventListener('change', renderJudgeList);
      document.getElementById('judgeDatasetFilter').addEventListener('change', renderJudgeList);
      document.getElementById('judgePromptFilter').addEventListener('change', renderJudgeList);
      renderJudgeList();
    }

    function renderJudgeList() {
      const judgeModelF  = document.getElementById('judgeModelFilter').value;
      const srcModelF    = document.getElementById('sourceModelFilter').value;
      const judgeModeF   = document.getElementById('judgeModeFilter').value;
      const datasetF     = document.getElementById('judgeDatasetFilter').value;
      const promptF      = document.getElementById('judgePromptFilter').value;

      const filtered = allJudgeTrials.filter(t =>
        (judgeModelF === 'all' || t.judge_model   === judgeModelF) &&
        (srcModelF   === 'all' || t.source_model  === srcModelF)   &&
        (judgeModeF  === 'all' || t.judge_mode    === judgeModeF)  &&
        (datasetF    === 'all' || t.dataset        === datasetF)   &&
        (promptF     === 'all' || t.prompt_type    === promptF)
      );

      document.getElementById('judgeTrialCount').textContent = `${filtered.length} trial${filtered.length !== 1 ? 's' : ''}`;
      document.getElementById('judgeTrialList').innerHTML = filtered.map(t => {
        const scoreColor = parseFloat(t.overall_score) >= 4 ? '#e8f5e9' : parseFloat(t.overall_score) >= 2.5 ? '#fff8e1' : '#ffebee';
        return `
        <div class="trial-row ${t.judge_trial_id === activeJudgeTrialId ? 'active' : ''}" onclick="showJudgeTrial('${t.judge_trial_id}')">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>${escapeHtml(labelPrompt(t.prompt_type))}</strong>
            ${t.overall_score ? `<span class="pill" style="background:${scoreColor};">Score: ${escapeHtml(t.overall_score)}</span>` : ''}
          </div>
          <div style="font-size:12px;color:#5a4e3a;">${escapeHtml(labelDataset(t.dataset))} &nbsp;·&nbsp; ${escapeHtml(labelModel(t.source_model))} &nbsp;·&nbsp; ${escapeHtml(t.judge_mode || '')}</div>
          <div style="font-size:12px;">run ${escapeHtml(t.run_id)} &nbsp;·&nbsp; query ${escapeHtml(t.query_index_within_episode)}</div>
          ${t.source_correct !== '' ? `<div class="${t.source_correct === '1' ? 'ok' : 'bad'}" style="font-size:12px;">${t.source_correct === '1' ? '✓ classifier correct' : '✗ classifier incorrect'}</div>` : ''}
          ${t.judge_parse_error ? `<div class="warn" style="font-size:11px;">⚠ parse error</div>` : ''}
        </div>`;
      }).join('');
    }

    async function showJudgeTrial(judgeTrialId) {
      activeJudgeTrialId = judgeTrialId;
      renderJudgeList();
      const detail = await fetch(`/api/judge-trial?id=${judgeTrialId}`).then(r => r.json());
      const meta = detail.metadata;
      const classIdMap = detail.class_id_map || {};
      const src = detail.source_trial || {};
      const srcCorrect = src.correct;
      const overall = meta.overall_score || '?';

      const scoresHtml = `
        <details class="section" open>
          <summary>Scores &nbsp;<span class="pill" style="background:#e3f2fd;">Overall: ${escapeHtml(overall)}</span></summary>
          <div class="section-body">
            <div class="stat-grid" style="grid-template-columns: repeat(5, 1fr);">${renderScoresGrid(meta)}</div>
          </div>
        </details>`;

      const judgeConvHtml = `
        <details class="section" open>
          <summary>What the judge received</summary>
          <div class="section-body">
            ${renderMessages(detail.judge_messages)}
          </div>
        </details>`;

      const parsedJudge = detail.parsed_judge_response && detail.parsed_judge_response.length
        ? detail.parsed_judge_response
            .filter(b => !b.tag.endsWith('_reasoning'))
            .map(b => `<div style="margin-bottom:10px;"><strong>&lt;${escapeHtml(b.tag)}&gt;</strong><pre style="margin-top:6px;">${escapeHtml(b.content)}</pre></div>`)
            .join('')
        : '';

      const reasoningBlocks = detail.parsed_judge_response
        ? detail.parsed_judge_response.filter(b => b.tag.endsWith('_reasoning'))
        : [];
      const reasoningHtml = reasoningBlocks.length ? `
        <details class="section">
          <summary>Score reasoning (per dimension)</summary>
          <div class="section-body">
            ${reasoningBlocks.map(b => `<div style="margin-bottom:10px;"><strong>${escapeHtml(b.tag.replace('_reasoning', ''))}</strong><pre style="margin-top:4px;font-size:0.9em;">${escapeHtml(b.content)}</pre></div>`).join('')}
          </div>
        </details>` : '';

      document.getElementById('judgeTrialDetail').innerHTML = `
        <details class="section" open>
          <summary>Judge Metadata</summary>
          <div class="section-body">
            <table>
              <tr><th>Judge Model</th><td>${escapeHtml(meta.judge_model || '')}</td></tr>
              <tr><th>Judge Mode</th><td>${escapeHtml(meta.judge_mode || '')}</td></tr>
              <tr><th>Source Model</th><td>${escapeHtml(meta.source_model || '')}</td></tr>
              <tr><th>Condition</th><td>${escapeHtml(labelPrompt(meta.prompt_type || ''))}</td></tr>
              <tr><th>Dataset</th><td>${escapeHtml(labelDataset(meta.dataset || ''))}</td></tr>
              <tr><th>Config</th><td>${escapeHtml(meta.config_n)}‑way &nbsp;${escapeHtml(meta.config_k)}‑shot</td></tr>
              <tr><th>Run / Query</th><td>run ${escapeHtml(meta.run_id)} &nbsp;·&nbsp; query ${escapeHtml(meta.query_index_within_episode)}</td></tr>
              ${srcCorrect !== undefined ? `<tr><th>Classifier result</th><td class="${srcCorrect === '1' ? 'ok' : 'bad'}">${srcCorrect === '1' ? '✓ correct' : '✗ incorrect'}</td></tr>` : ''}
              ${meta.latency_seconds ? `<tr><th>Judge latency</th><td>${escapeHtml(meta.latency_seconds)} s</td></tr>` : ''}
              ${meta.usage_total_tokens ? `<tr><th>Judge tokens</th><td>${escapeHtml(meta.usage_total_tokens)}</td></tr>` : ''}
              ${meta.judge_parse_error ? `<tr><th>Parse error</th><td class="warn">${escapeHtml(meta.judge_parse_error)}</td></tr>` : ''}
            </table>
          </div>
        </details>
        ${scoresHtml}
        ${reasoningHtml}
        ${judgeConvHtml}
        ${parsedJudge ? `<details class="section" open><summary>Judge Response (parsed)</summary><div class="section-body">${parsedJudge}</div></details>` : ''}
        <details class="section">
          <summary>📝 Full Judge Output (raw)</summary>
          <div class="section-body"><pre style="font-size:0.9em;background:#fffde7;border:1px solid #ffe082;">${escapeHtml(detail.judge_raw_response || '')}</pre></div>
        </details>
      `;
    }

    // ── Bootstrap ──────────────────────────────────────────────────
    async function bootstrap() {
      await loadSummary();
      await loadTrials();
      if (allTrials.length) showTrial(allTrials[0].trial_id);
    }
    bootstrap();
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    store: RunDataStore

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_dashboard_html())
            return
        if parsed.path == "/api/summary":
            self._send_json(self.store.summary_payload())
            return
        if parsed.path == "/api/trials":
            self._send_json(self.store.list_trials())
            return
        if parsed.path == "/api/trial":
            query = parse_qs(parsed.query)
            trial_id = query.get("trial_id", ["0"])[0]
            self._send_json(self.store.get_trial(trial_id))
            return
        if parsed.path == "/api/judge-trials":
            self._send_json(self.store.list_judge_trials())
            return
        if parsed.path == "/api/judge-trial":
            query = parse_qs(parsed.query)
            judge_trial_id = query.get("id", ["0"])[0]
            self._send_json(self.store.get_judge_trial(judge_trial_id))
            return
        if parsed.path == "/api/image":
            query = parse_qs(parsed.query)
            dataset = query.get("dataset", [None])[0]
            index = query.get("index", [None])[0]
            if dataset is None or index is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "dataset and index are required")
                return
            image_bytes = self.store.load_image_bytes(dataset, int(index))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image_bytes)))
            self.end_headers()
            self.wfile.write(image_bytes)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


@dataclass
class DashboardServerHandle:
    server: ThreadingHTTPServer
    url: str


def build_dashboard_server(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> DashboardServerHandle:
    store = RunDataStore(run_dir)

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.store = store
    server = ThreadingHTTPServer((host, port), BoundHandler)
    return DashboardServerHandle(server=server, url=f"http://{host}:{port}/")


def serve_dashboard(run_dir: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    handle = build_dashboard_server(run_dir, host=host, port=port)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(handle.url)).start()
    print(f"[*] Dashboard serving {run_dir} at {handle.url}")
    print("[*] Press Ctrl+C to stop the server.")
    try:
        handle.server.serve_forever()
    finally:
        handle.server.server_close()
