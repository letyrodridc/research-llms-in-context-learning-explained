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

        # Load Judge Results if they exist
        self.judge_data = {}
        judge_dir = self.run_dir / "judge_outputs"
        if judge_dir.exists():
            for judge_file in judge_dir.glob("judge_results_*.csv"):
                j_rows = _read_csv_safe(judge_file)
                for jr in j_rows:
                    # Key by dataset+prompt_type+run_id+query_index
                    key = f"{jr['dataset']}_{jr['prompt_type']}_{jr['run_id']}_{jr['query_index']}"
                    self.judge_data[key] = jr

        for idx, row in enumerate(self.trial_rows):
            row["_trial_id"] = str(idx)
            # Link judge data to trial row
            key = f"{row['dataset']}_{row['prompt_type']}_{row['run_id']}_{row['query_index_within_episode']}"
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
    .layout { display: grid; grid-template-columns: 440px 1fr; min-height: 100vh; }
    .sidebar { background: #fffaf1; border-right: 1px solid #d9d0bf; padding: 16px; overflow: auto; display: flex; flex-direction: column; gap: 10px; }
    .main { padding: 20px; overflow: auto; }
    h2, h3 { margin: 0 0 10px; }
    .card { background: white; border: 1px solid #d9d0bf; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }

    /* collapsible sections */
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
    .filter-model { grid-column: 1 / -1; }
    .trial-count { font-size: 12px; color: #7a6e5e; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ede4d6; font-size: 13px; }
    th { background: #f7f2e8; font-weight: 600; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #efe6d3; margin-right: 4px; font-size: 12px; }
    .run-meta { font-size: 13px; }
    .run-meta strong { font-size: 15px; display: block; margin-bottom: 4px; }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .stat { background: #fff7e3; border-radius: 10px; padding: 10px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <div id="runMeta" class="card run-meta"></div>

      <details class="section" open>
        <summary>Trials</summary>
        <div class="section-body">
          <div class="filters">
            <div class="filter-model">
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

  <script>
    const PROMPT_LABELS = {
      'classification':       'Classification without explanation',
      'nle':                  'Natural language explanation',
      'features':             'Features-based explanation',
      'rulebased':            'Feature-value based explanations',
      'axioms_ontology_v2':   'DL Axioms',
    };
    const DATASET_LABELS = {
      'flowers': 'Flowers 102',
      'pets':    'Oxford Pets',
      'cifar10': 'CIFAR-10',
      'dtd':     'DTD',
    };
    function labelPrompt(v) { return PROMPT_LABELS[v] || v; }
    function labelDataset(v) { return DATASET_LABELS[v] || v; }
    function labelModel(v) { return v ? v.split('/').pop() : v; }

    let allTrials = [];
    let activeTrialId = null;

    function escapeHtml(value) {
      return (value ?? '').toString()
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    async function loadSummary() {
      const summary = await fetch('/api/summary').then(r => r.json());
      document.getElementById('runMeta').innerHTML = `
        <strong>${escapeHtml(summary.run_name)}</strong>
        <div style="color:#7a6e5e;font-size:12px;">${escapeHtml(summary.run_dir)}</div>
        <div style="margin-top:8px;"><span class="pill">Trials: ${summary.trial_count}</span></div>
      `;

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
      const modelFilter   = document.getElementById('modelFilter').value;
      const datasetFilter = document.getElementById('datasetFilter').value;
      const promptFilter  = document.getElementById('promptFilter').value;

      const filtered = allTrials.filter(trial =>
        (modelFilter   === 'all' || trial.model        === modelFilter)   &&
        (datasetFilter === 'all' || trial.dataset      === datasetFilter) &&
        (promptFilter  === 'all' || trial.prompt_type  === promptFilter)
      );

      document.getElementById('trialCount').textContent = `${filtered.length} trial${filtered.length !== 1 ? 's' : ''}`;
      document.getElementById('trialList').innerHTML = filtered.map(trial => `
        <div class="trial-row ${trial.trial_id === activeTrialId ? 'active' : ''}" onclick="showTrial('${trial.trial_id}')">
          <div style="display:flex; justify-content: space-between; align-items: center;">
            <strong>${escapeHtml(labelPrompt(trial.prompt_type))}</strong>
            ${trial.judge_scores ? `<span class="pill" style="background:#e3f2fd;">Judge: ${escapeHtml(trial.judge_scores.visual_grounding)}/5</span>` : ''}
          </div>
          <div style="font-size:12px;color:#5a4e3a;">${escapeHtml(labelDataset(trial.dataset))} &nbsp;·&nbsp; ${escapeHtml(labelModel(trial.model))}</div>
          <div style="font-size:12px;">run ${escapeHtml(trial.run_id)} &nbsp;·&nbsp; expected: <em>${escapeHtml(trial.expected_name)}</em> &nbsp;·&nbsp; predicted: <em>${escapeHtml(trial.predicted_name)}</em></div>
          <div class="${trial.correct === '1' ? 'ok' : 'bad'}">${trial.correct === '1' ? '✓ correct' : '✗ incorrect'}</div>
          ${(trial.parse_issue || trial.warning || trial.error) ? `<div class="warn">⚠ ${escapeHtml(trial.parse_issue || trial.warning || trial.error)}</div>` : ''}
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

      const messages = detail.conversation.map(message => {
        if (typeof message.content === 'string') {
          return `
            <div class="message">
              <div class="message-header">${escapeHtml(message.role)}</div>
              <div class="message-body"><pre>${escapeHtml(message.content)}</pre></div>
            </div>`;
        }
        const parts = message.content.map(part => {
          if (part.type === 'text') {
            return `<div class="content-part"><pre>${escapeHtml(part.text)}</pre></div>`;
          }
          if (part.type === 'image_ref') {
            const kind = part.image_ref.kind;
            return `<div class="content-part img-wrap">
              <div class="img-label ${kind}">${escapeHtml(kind)}</div>
              <img class="preview ${kind}-img" src="${part.image_ref.url}" />
            </div>`;
          }
          return `<div class="content-part"><pre>${escapeHtml(JSON.stringify(part, null, 2))}</pre></div>`;
        }).join('');
        return `
          <div class="message">
            <div class="message-header">${escapeHtml(message.role)}</div>
            <div class="message-body">${parts}</div>
          </div>`;
      }).join('');

      const parsed = detail.parsed_response.length
        ? detail.parsed_response.map(b => `<div style="margin-bottom:10px;"><strong>&lt;${escapeHtml(b.tag)}&gt;</strong><pre style="margin-top:6px;">${escapeHtml(b.content)}</pre></div>`).join('')
        : '<div style="color:#7a6e5e;">No XML blocks parsed from the model response.</div>';

      const promptDisplay = labelPrompt(meta.prompt_type || '');
      const datasetDisplay = labelDataset(meta.dataset || '');
      const modelDisplay = labelModel(meta.model || '');

      const judgeHtml = detail.judge_scores ? `
        <details class="section" open>
          <summary>👨‍⚖️ Judge Evaluation</summary>
          <div class="section-body">
            <div class="stat-grid">
              <div class="stat"><strong>Visual Grounding</strong><br/>${escapeHtml(detail.judge_scores.visual_grounding)}/5</div>
              <div class="stat"><strong>Discriminative</strong><br/>${escapeHtml(detail.judge_scores.discriminative_support)}/5</div>
              <div class="stat"><strong>Coherence</strong><br/>${escapeHtml(detail.judge_scores.inferential_coherence)}/5</div>
              <div class="stat"><strong>Clarity</strong><br/>${escapeHtml(detail.judge_scores.clarity)}/5</div>
              <div class="stat"><strong>Compliance</strong><br/>${escapeHtml(detail.judge_scores.format_compliance)}/5</div>
            </div>
            <div style="margin-top:12px;">
              <strong>Full Critique:</strong>
              <pre style="font-size:0.9em; max-height: 200px; overflow-y: auto;">${escapeHtml(detail.judge_scores.raw_judge_output)}</pre>
            </div>
          </div>
        </details>` : '';

      document.getElementById('trialDetail').innerHTML = `
        ${judgeHtml}
        <details class="section" open>
          <summary>Trial Metadata</summary>
          <div class="section-body">
            <table>
              <tr><th>Condition</th><td>${escapeHtml(promptDisplay)}</td></tr>
              <tr><th>Dataset</th><td>${escapeHtml(datasetDisplay)}</td></tr>
              <tr><th>Model</th><td>${escapeHtml(modelDisplay)}</td></tr>
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
