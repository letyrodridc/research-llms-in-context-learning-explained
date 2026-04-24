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

from .prompt_assets import repo_root
from .prompts import _tensor_to_pil
from .reconstruction import reconstruct_classifier_messages


def _read_csv(path: Path) -> List[Dict[str, str]]:
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
        self.trial_rows = _read_csv(self.run_dir / "trial_results.csv")
        self.summary_rows = _read_csv(self.run_dir / "experiment_summary.csv")
        self.run_rows = _read_csv(self.run_dir / "run_accuracy_long.csv")
        self.experiment_config = _read_json_if_exists(self.run_dir / "experiment_config.json")
        self.experiment_snapshot = _read_json_if_exists(self.run_dir / "experiment_config_snapshot.json")
        self.prompt_library = _read_json_if_exists(self.run_dir / "prompt_library_snapshot.json")
        self.run_manifest = _read_json_if_exists(self.run_dir / "run_manifest.json")
        self._datasets_cache: Optional[Dict[str, Any]] = None

        for idx, row in enumerate(self.trial_rows):
            row["_trial_id"] = str(idx)

    def _load_datasets(self) -> Dict[str, Any]:
        if self._datasets_cache is None:
            from setup_utils import load_datasets

            self._datasets_cache = load_datasets(data_dir=str(self.repo_root / "data"))
        return self._datasets_cache

    def list_trials(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for row in self.trial_rows:
            output.append(
                {
                    "trial_id": row["_trial_id"],
                    "dataset": row["dataset"],
                    "prompt_type": row["prompt_type"],
                    "config_n": row["config_n"],
                    "config_k": row["config_k"],
                    "config_q": row["config_q"],
                    "run_id": row["run_id"],
                    "query_index_within_episode": row["query_index_within_episode"],
                    "expected_label": row["expected_label"],
                    "predicted_label": row["predicted_label"],
                    "correct": row["correct"],
                    "parse_issue": row.get("parse_issue", ""),
                    "warning": row.get("warning", ""),
                    "error": row.get("error", ""),
                    "latency_seconds": row.get("latency_seconds", ""),
                    "usage_total_tokens": row.get("usage_total_tokens", ""),
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
            "class_options": _parse_json_field(row.get("class_options", ""), []),
            "image_refs": _parse_json_field(row.get("image_refs", ""), []),
            "conversation": _browser_messages(
                reconstructed.classifier_messages,
                reconstructed.image_refs,
                dataset_name,
            ),
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
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f5f2ea; color: #1f2430; }
    .layout { display: grid; grid-template-columns: 420px 1fr; min-height: 100vh; }
    .sidebar { background: #fffaf1; border-right: 1px solid #d9d0bf; padding: 16px; overflow: auto; }
    .main { padding: 20px; overflow: auto; }
    h1,h2,h3 { margin: 0 0 12px; }
    .card { background: white; border: 1px solid #d9d0bf; border-radius: 12px; padding: 14px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .stat { background: #fff7e3; border-radius: 10px; padding: 10px; }
    .trial-row { padding: 10px; border-top: 1px solid #ece4d7; cursor: pointer; }
    .trial-row:hover { background: #fff7e3; }
    .trial-row.active { background: #ffe8b5; }
    .ok { color: #0a7f3f; }
    .bad { color: #a12a2a; }
    .warn { color: #8b5e00; }
    pre { white-space: pre-wrap; word-break: break-word; background: #faf7f0; padding: 10px; border-radius: 8px; border: 1px solid #e5dccb; }
    .message { border: 1px solid #ddd1be; border-radius: 12px; margin-bottom: 12px; background: white; overflow: hidden; }
    .message-header { padding: 8px 12px; font-weight: 700; background: #f1eadc; }
    .message-body { padding: 12px; }
    .content-part { margin-bottom: 10px; }
    img.preview { max-width: 280px; max-height: 280px; border-radius: 10px; border: 1px solid #d8cdb7; display: block; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 10px; }
    select, button { padding: 8px 10px; border-radius: 8px; border: 1px solid #c9bea9; background: white; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ede4d6; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #efe6d3; margin-right: 6px; }
  </style>
</head>
<body>
  <div class="layout">
    <div class="sidebar">
      <h2>Run Dashboard</h2>
      <div id="runMeta" class="card"></div>
      <div class="card">
        <div class="toolbar">
          <select id="datasetFilter"></select>
          <select id="promptFilter"></select>
        </div>
        <div id="trialList"></div>
      </div>
    </div>
    <div class="main">
      <div id="summaryCard" class="card"></div>
      <div id="configCard" class="card"></div>
      <div id="trialDetail"></div>
    </div>
  </div>
  <script>
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
        <div><strong>${escapeHtml(summary.run_name)}</strong></div>
        <div>${escapeHtml(summary.run_dir)}</div>
        <div style="margin-top:8px;"><span class="pill">Trials: ${summary.trial_count}</span></div>
      `;

      const rows = summary.experiment_summary.map(row => `
        <tr>
          <td>${escapeHtml(row.prompt_type)}</td>
          <td>${escapeHtml(row.overall_accuracy)}</td>
          <td>${escapeHtml(row.total_trials)}</td>
          <td>${escapeHtml(row.total_errors)}</td>
        </tr>
      `).join('');
      document.getElementById('summaryCard').innerHTML = `
        <h3>Experiment Summary</h3>
        <table>
          <thead><tr><th>Prompt</th><th>Accuracy</th><th>Trials</th><th>Errors</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;

      document.getElementById('configCard').innerHTML = `
        <h3>Config Snapshot</h3>
        <pre>${escapeHtml(JSON.stringify(summary.experiment_config || summary.experiment_snapshot || {}, null, 2))}</pre>
      `;
    }

    async function loadTrials() {
      allTrials = await fetch('/api/trials').then(r => r.json());
      const datasets = ['all', ...new Set(allTrials.map(t => t.dataset))];
      const prompts = ['all', ...new Set(allTrials.map(t => t.prompt_type))];
      document.getElementById('datasetFilter').innerHTML = datasets.map(v => `<option value="${v}">${v}</option>`).join('');
      document.getElementById('promptFilter').innerHTML = prompts.map(v => `<option value="${v}">${v}</option>`).join('');
      document.getElementById('datasetFilter').addEventListener('change', renderTrialList);
      document.getElementById('promptFilter').addEventListener('change', renderTrialList);
      renderTrialList();
    }

    function renderTrialList() {
      const datasetFilter = document.getElementById('datasetFilter').value;
      const promptFilter = document.getElementById('promptFilter').value;
      const filtered = allTrials.filter(trial =>
        (datasetFilter === 'all' || trial.dataset === datasetFilter) &&
        (promptFilter === 'all' || trial.prompt_type === promptFilter)
      );
      document.getElementById('trialList').innerHTML = filtered.map(trial => `
        <div class="trial-row ${trial.trial_id === activeTrialId ? 'active' : ''}" onclick="showTrial('${trial.trial_id}')">
          <div><strong>${escapeHtml(trial.prompt_type)}</strong> | ${escapeHtml(trial.dataset)}</div>
          <div>run=${trial.run_id} q=${trial.query_index_within_episode} expected=${escapeHtml(trial.expected_label)} predicted=${escapeHtml(trial.predicted_label)}</div>
          <div class="${trial.correct === '1' ? 'ok' : 'bad'}">${trial.correct === '1' ? 'correct' : 'incorrect'}</div>
          ${(trial.parse_issue || trial.warning || trial.error) ? `<div class="warn">${escapeHtml(trial.parse_issue || trial.warning || trial.error)}</div>` : ''}
        </div>
      `).join('');
    }

    async function showTrial(trialId) {
      activeTrialId = trialId;
      renderTrialList();
      const detail = await fetch(`/api/trial?trial_id=${trialId}`).then(r => r.json());
      const messages = detail.conversation.map(message => {
        if (typeof message.content === 'string') {
          return `
            <div class="message">
              <div class="message-header">${escapeHtml(message.role)}</div>
              <div class="message-body"><pre>${escapeHtml(message.content)}</pre></div>
            </div>
          `;
        }
        const parts = message.content.map(part => {
          if (part.type === 'text') {
            return `<div class="content-part"><pre>${escapeHtml(part.text)}</pre></div>`;
          }
          if (part.type === 'image_ref') {
            return `<div class="content-part">
              <div><span class="pill">${escapeHtml(part.image_ref.kind)}</span><span class="pill">idx=${part.image_ref.dataset_index}</span></div>
              <img class="preview" src="${part.image_ref.url}" />
            </div>`;
          }
          return `<div class="content-part"><pre>${escapeHtml(JSON.stringify(part, null, 2))}</pre></div>`;
        }).join('');
        return `
          <div class="message">
            <div class="message-header">${escapeHtml(message.role)}</div>
            <div class="message-body">${parts}</div>
          </div>
        `;
      }).join('');

      const parsed = detail.parsed_response.length
        ? detail.parsed_response.map(block => `<div class="card"><h3>&lt;${escapeHtml(block.tag)}&gt;</h3><pre>${escapeHtml(block.content)}</pre></div>`).join('')
        : '<div class="card"><h3>Parsed XML</h3><div>No XML blocks parsed from the model response.</div></div>';

      document.getElementById('trialDetail').innerHTML = `
        <div class="card">
          <h3>Trial Metadata</h3>
          <pre>${escapeHtml(JSON.stringify(detail.metadata, null, 2))}</pre>
        </div>
        <div class="card">
          <h3>Conversation</h3>
          ${messages}
        </div>
        ${parsed}
      `;
    }

    async function bootstrap() {
      await loadSummary();
      await loadTrials();
      if (allTrials.length) {
        showTrial(allTrials[0].trial_id);
      }
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
