from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline.experiments.analysis import analyze_run_directory
from pipeline.utils.client import OpenRouterClient, extract_message_text, model_supports_images
from pipeline.experiments.config import OpenRouterSettings
from pipeline.dashboard.dashboard import RunDataStore
from pipeline.experiments.experiment_config import export_experiment_config_snapshot, load_experiment_config
from pipeline.evaluation.judge_analysis import analyze_judge_run_directory
from pipeline.evaluation.judge_prompts import JUDGE_PROMPT_SPECS
from pipeline.utils.prompt_assets import repo_relative_path, resolve_repo_path
from pipeline.experiments.prompts import PROMPT_SPECS, pil_image_to_data_url
from pipeline.experiments.run_openrouter_experiment import build_trial_record_base, sanitize_messages_for_logging


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.post_calls = []

    def post(self, url, headers, json, timeout):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(
            {
                "id": "resp_123",
                "provider": {"name": "mock"},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<response>Bombay</response>"},
                    }
                ],
            }
        )

    def get(self, url, headers, timeout):
        return FakeResponse(
            {
                "data": [
                    {
                        "id": "test/model",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            }
        )


class OpenRouterModeTests(unittest.TestCase):
    def test_extract_message_text_handles_text_blocks(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(extract_message_text(payload), "Hello\nWorld")

    def test_pil_image_to_data_url_uses_base64_data_uri(self):
        image = Image.new("RGB", (4, 4), color="red")
        data_url = pil_image_to_data_url(image)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(data_url), 32)

    def test_prompt_specs_are_loaded_from_asset_files(self):
        self.assertIn(
            "Provide only the final class label inside the response tag.",
            PROMPT_SPECS["classification"].system_prompt,
        )
        self.assertIn("<explanation>", PROMPT_SPECS["nle"].system_prompt)
        self.assertIn("Visual Grounding", JUDGE_PROMPT_SPECS["nle"].system_prompt)

    def test_experiment_config_loads_prompt_library_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_library_path = tmp_path / "prompt_library.json"
            prompt_library_path.write_text(
                """
{
  "schema_version": "openrouter_prompt_library_v1",
  "name": "tmp",
  "shared_system_prompt": "Header\\n\\n{CONDITION_INSTRUCTION}\\n\\n<response>label</response>",
  "query_template": "Options: [{OPTIONS}]",
  "prompt_types": {
    "classification": {"condition_instruction": "Return only the label.", "max_tokens": 32, "generation": {"temperature": 0.0}},
    "nle": {"condition_instruction": "<explanation>why</explanation>", "max_tokens": 64, "generation": {"temperature": 0.0}},
    "features": {"condition_instruction": "<features>- x</features>", "max_tokens": 64, "generation": {"temperature": 0.0}},
    "rulebased": {"condition_instruction": "<kb>- rule</kb>", "max_tokens": 64, "generation": {"temperature": 0.0}},
    "axioms_ontology_v2": {"condition_instruction": "<tbox>- axiom</tbox>", "max_tokens": 64, "generation": {"temperature": 0.0}}
  }
}
                """.strip(),
                encoding="utf-8",
            )
            config_path = tmp_path / "experiment.json"
            config_path.write_text(
                f"""
{{
  "schema_version": "openrouter_experiment_v2",
  "experiment_name": "tmp-run",
  "env_file": ".env",
  "output_root": "outputs",
  "prompt_library_path": "{prompt_library_path.name}",
  "model": {{
    "model_name": ["test/model", "backup/model"],
    "model_params": {{
      "validate_image_input": true,
      "generation": {{"temperature": 0.0}}
    }}
  }},
  "datasets": ["pets"],
  "prompt_types": ["classification", "nle"],
  "few_shot_configs": [{{"n": 2, "k": 1, "q": 3, "runs": 1}}],
  "seed": 7
}}
                """.strip(),
                encoding="utf-8",
            )

            loaded = load_experiment_config(config_path)

            self.assertEqual(loaded.experiment_name, "tmp-run")
            self.assertEqual(loaded.model.name, "test/model")
            self.assertEqual(loaded.model.names, ["test/model", "backup/model"])
            self.assertEqual(loaded.datasets, ["pets"])
            self.assertEqual(loaded.prompt_types, ["classification", "nle"])
            self.assertEqual(loaded.few_shot_configs[0].n, 2)
            self.assertEqual(loaded.few_shot_configs[0].runs, 1)
            self.assertEqual(loaded.output_root, (tmp_path / "outputs").resolve())
            self.assertIn("Return only the label.", loaded.prompt_library.prompt_specs["classification"].system_prompt)

    def test_sanitize_messages_for_logging_replaces_data_urls_with_image_refs(self):
        messages = [
            {"role": "system", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                ],
            },
        ]
        image_refs = [{"kind": "query", "dataset_index": 10, "order_index": 1}]

        sanitized = sanitize_messages_for_logging(messages, image_refs)

        self.assertEqual(sanitized[0]["content"], "hello")
        self.assertEqual(sanitized[1]["content"][0]["text"], "What is this?")
        self.assertEqual(sanitized[1]["content"][1]["type"], "image_ref")
        self.assertEqual(sanitized[1]["content"][1]["image_ref"]["dataset_index"], 10)

    def test_export_experiment_config_snapshot_uses_repo_relative_paths(self):
        loaded = load_experiment_config(REPO_ROOT / "pipeline" / "configs" / "openrouter_experiment.test.json")

        snapshot = export_experiment_config_snapshot(loaded)

        self.assertEqual(snapshot["source_config"]["path"], "pipeline\\configs\\openrouter_experiment.test.json")
        self.assertEqual(snapshot["resolved"]["env_file"], ".env")
        self.assertEqual(snapshot["resolved"]["output_root"], "pipeline\\openrouter_runs")

    def test_build_trial_record_base_stores_repo_relative_episode_path(self):
        episode_path = REPO_ROOT / "episodes" / "seed_42" / "pets" / "episode_N2_K1_Q1_run0.npy"
        artifact_dir = REPO_ROOT / "pipeline" / "openrouter_runs" / "demo" / "datasets" / "pets" / "classification" / "N2_K1_Q1" / "run_0"
        run_dir = REPO_ROOT / "pipeline" / "openrouter_runs" / "demo"
        messages = [{"role": "system", "content": "hello"}]

        record = build_trial_record_base(
            run_id=0,
            dataset_name="pets",
            prompt_type="classification",
            config_tuple=type("Cfg", (), {"n": 2, "k": 1, "q": 1})(),
            support_indices=[1, 2],
            query_dataset_index=10,
            query_index_within_episode=0,
            expected_label="Bombay",
            episode_filepath=episode_path,
            class_options=["Bombay", "Siamese"],
            messages=messages,
            model_name="test/model",
            artifact_dir=artifact_dir,
            run_dir=run_dir,
        )

        self.assertEqual(record["episode_filepath"], repo_relative_path(episode_path))

    def test_resolve_repo_path_remaps_old_absolute_repo_paths(self):
        old_machine_path = Path(
            "C:/Users/Someone/Documents/GitHub/research-llms-in-context-learning-explained/README.md"
        )

        resolved = resolve_repo_path(old_machine_path)

        self.assertEqual(resolved, (REPO_ROOT / "README.md").resolve())

    def test_dashboard_store_lists_trials_without_loading_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            trial_rows = [
                {
                    "trial_timestamp": "2026-03-20T12:00:00",
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "query_index_within_episode": "0",
                    "support_indices": "[1,2]",
                    "query_dataset_index": "10",
                    "expected_label": "Bombay",
                    "predicted_label": "Bombay",
                    "correct": "1",
                    "error": "",
                    "warning": "",
                    "parse_issue": "",
                    "system_fallback_applied": "0",
                    "trial_wall_seconds": "0.5",
                    "latency_seconds": "0.5",
                    "response_id": "a",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "10",
                    "usage_completion_tokens": "5",
                    "usage_total_tokens": "15",
                    "provider": "{}",
                    "episode_filepath": "episode.npy",
                    "class_options": "[\"Bombay\"]",
                    "image_refs": "[]",
                    "prompt_hash": "abc",
                    "message_preview": "[]",
                    "sent_prompt_hash": "abc",
                    "sent_message_preview": "[]",
                    "artifact_dir": "datasets/pets/classification/N2_K1_Q9/run_0",
                    "conversation_log_path": "datasets/pets/classification/N2_K1_Q9/run_0/conversations.jsonl",
                    "raw_response_text": "<response>Bombay</response>",
                }
            ]
            run_rows = [
                {
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "correct": "1",
                    "total": "1",
                    "errors": "0",
                    "accuracy": "1.0000",
                    "run_duration_seconds": "0.5",
                    "avg_trial_wall_seconds": "0.5",
                    "avg_trial_api_seconds": "0.4",
                    "system_fallback_count": "0",
                    "artifact_dir": "datasets/pets/classification/N2_K1_Q9/run_0",
                }
            ]
            summary_rows = [
                {
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "total_correct": "1",
                    "total_trials": "1",
                    "total_errors": "0",
                    "overall_accuracy": "1.0000",
                    "total_duration_seconds": "0.5",
                    "avg_trial_wall_seconds": "0.5",
                    "avg_trial_api_seconds": "0.4",
                    "system_fallback_count": "0",
                }
            ]

            for filename, rows in [
                ("trial_results.csv", trial_rows),
                ("run_accuracy_long.csv", run_rows),
                ("experiment_summary.csv", summary_rows),
            ]:
                with (run_dir / filename).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

            (run_dir / "run_manifest.json").write_text('{"run_name":"demo"}', encoding="utf-8")

            store = RunDataStore(run_dir)
            trials = store.list_trials()
            summary = store.summary_payload()

            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0]["trial_id"], "0")
            self.assertEqual(summary["trial_count"], 1)

    def test_openrouter_client_parses_openai_compatible_response(self):
        session = FakeSession()
        settings = OpenRouterSettings(
            api_key="test-key",
            model="test/model",
            site_url="https://example.com",
            app_name="unit-test",
            timeout_seconds=30,
            max_retries=2,
        )
        client = OpenRouterClient(settings, session=session)
        response = client.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=128)

        self.assertEqual(response.text, "<response>Bombay</response>")
        self.assertEqual(response.request_id, "resp_123")
        self.assertEqual(response.usage["total_tokens"], 15)
        self.assertEqual(session.post_calls[0]["json"]["model"], "test/model")
        self.assertEqual(session.post_calls[0]["json"]["max_tokens"], 128)

    def test_model_supports_images_reads_metadata(self):
        self.assertTrue(
            model_supports_images({"architecture": {"input_modalities": ["text", "image"]}})
        )
        self.assertFalse(model_supports_images({"architecture": {"input_modalities": ["text"]}}))

    def test_analyze_run_directory_generates_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)

            trial_rows = [
                {
                    "trial_timestamp": "2026-03-20T12:00:00",
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "query_index_within_episode": "0",
                    "support_indices": "[1,2]",
                    "query_dataset_index": "10",
                    "expected_label": "Bombay",
                    "predicted_label": "Bombay",
                    "correct": "1",
                    "error": "",
                    "latency_seconds": "0.5",
                    "response_id": "a",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "10",
                    "usage_completion_tokens": "5",
                    "usage_total_tokens": "15",
                    "provider": "{}",
                    "episode_filepath": "episode.npy",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "prompt_hash": "abc",
                    "message_preview": "[]",
                    "raw_response_text": "<response>Bombay</response>",
                },
                {
                    "trial_timestamp": "2026-03-20T12:00:01",
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "query_index_within_episode": "0",
                    "support_indices": "[1,2]",
                    "query_dataset_index": "10",
                    "expected_label": "Bombay",
                    "predicted_label": "Bombay",
                    "correct": "1",
                    "error": "",
                    "latency_seconds": "0.4",
                    "response_id": "b",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "10",
                    "usage_completion_tokens": "5",
                    "usage_total_tokens": "15",
                    "provider": "{}",
                    "episode_filepath": "episode.npy",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "prompt_hash": "def",
                    "message_preview": "[]",
                    "raw_response_text": "<response>Bombay</response>",
                },
                {
                    "trial_timestamp": "2026-03-20T12:00:02",
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "query_index_within_episode": "0",
                    "support_indices": "[1,2]",
                    "query_dataset_index": "11",
                    "expected_label": "Bombay",
                    "predicted_label": "American Bulldog",
                    "correct": "0",
                    "error": "",
                    "latency_seconds": "0.6",
                    "response_id": "c",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "10",
                    "usage_completion_tokens": "5",
                    "usage_total_tokens": "15",
                    "provider": "{}",
                    "episode_filepath": "episode.npy",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "prompt_hash": "ghi",
                    "message_preview": "[]",
                    "raw_response_text": "<response>American Bulldog</response>",
                },
                {
                    "trial_timestamp": "2026-03-20T12:00:03",
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "query_index_within_episode": "0",
                    "support_indices": "[1,2]",
                    "query_dataset_index": "11",
                    "expected_label": "Bombay",
                    "predicted_label": "Bombay",
                    "correct": "1",
                    "error": "",
                    "latency_seconds": "0.4",
                    "response_id": "d",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "10",
                    "usage_completion_tokens": "5",
                    "usage_total_tokens": "15",
                    "provider": "{}",
                    "episode_filepath": "episode.npy",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "prompt_hash": "jkl",
                    "message_preview": "[]",
                    "raw_response_text": "<response>Bombay</response>",
                },
            ]

            run_rows = [
                {
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "correct": "1",
                    "total": "1",
                    "errors": "0",
                    "accuracy": "1.0000",
                },
                {
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "correct": "1",
                    "total": "1",
                    "errors": "0",
                    "accuracy": "1.0000",
                },
                {
                    "dataset": "pets",
                    "prompt_type": "classification",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "correct": "0",
                    "total": "1",
                    "errors": "0",
                    "accuracy": "0.0000",
                },
                {
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "model": "test/model",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "correct": "1",
                    "total": "1",
                    "errors": "0",
                    "accuracy": "1.0000",
                },
            ]

            with (run_dir / "trial_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trial_rows)

            with (run_dir / "run_accuracy_long.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
                writer.writeheader()
                writer.writerows(run_rows)

            outputs = analyze_run_directory(run_dir)

            self.assertTrue((run_dir / "analysis" / "tables" / "overall_accuracy_by_prompt.csv").exists())
            self.assertTrue((run_dir / "analysis" / "plots" / "overall_accuracy_by_prompt.png").exists())
            self.assertTrue((run_dir / "analysis" / "plots" / "overall_accuracy_by_prompt.json").exists())
            self.assertTrue(outputs["report_path"].exists())

    def test_analyze_judge_run_directory_generates_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            judge_rows = [
                {
                    "judge_timestamp": "2026-04-04T10:00:00",
                    "source_run_dir": "source-a",
                    "source_run_name": "source-a",
                    "source_model": "classifier/model",
                    "judge_model": "judge/model",
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "query_index_within_episode": "0",
                    "predicted_label": "Bombay",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "classifier_raw_response_text": "<explanation>Black short fur.</explanation><response>Bombay</response>",
                    "visual_grounding": "5",
                    "discriminative_support": "4",
                    "inferential_coherence": "4",
                    "clarity": "5",
                    "format_compliance": "5",
                    "overall_score": "4.6000",
                    "judge_parse_error": "",
                    "warning": "",
                    "trial_wall_seconds": "0.9",
                    "latency_seconds": "0.4",
                    "response_id": "jr1",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "20",
                    "usage_completion_tokens": "10",
                    "usage_total_tokens": "30",
                    "provider": "{}",
                    "source_prompt_hash": "abc",
                    "judge_prompt_hash": "def",
                    "judge_message_preview": "[]",
                    "judge_raw_response_text": "<evaluation></evaluation>",
                },
                {
                    "judge_timestamp": "2026-04-04T10:00:01",
                    "source_run_dir": "source-a",
                    "source_run_name": "source-a",
                    "source_model": "classifier/model",
                    "judge_model": "judge/model",
                    "dataset": "pets",
                    "prompt_type": "features",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "0",
                    "query_index_within_episode": "0",
                    "predicted_label": "Bombay",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "classifier_raw_response_text": "<features>- black fur</features><response>Bombay</response>",
                    "visual_grounding": "4",
                    "discriminative_support": "3",
                    "inferential_coherence": "4",
                    "clarity": "4",
                    "format_compliance": "5",
                    "overall_score": "4.0000",
                    "judge_parse_error": "",
                    "warning": "",
                    "trial_wall_seconds": "0.8",
                    "latency_seconds": "0.4",
                    "response_id": "jr2",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "20",
                    "usage_completion_tokens": "10",
                    "usage_total_tokens": "30",
                    "provider": "{}",
                    "source_prompt_hash": "ghi",
                    "judge_prompt_hash": "jkl",
                    "judge_message_preview": "[]",
                    "judge_raw_response_text": "<evaluation></evaluation>",
                },
                {
                    "judge_timestamp": "2026-04-04T10:00:02",
                    "source_run_dir": "source-a",
                    "source_run_name": "source-a",
                    "source_model": "classifier/model",
                    "judge_model": "judge/model",
                    "dataset": "pets",
                    "prompt_type": "nle",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "query_index_within_episode": "0",
                    "predicted_label": "Bombay",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "classifier_raw_response_text": "<explanation>Dark cat.</explanation><response>Bombay</response>",
                    "visual_grounding": "5",
                    "discriminative_support": "5",
                    "inferential_coherence": "5",
                    "clarity": "4",
                    "format_compliance": "5",
                    "overall_score": "4.8000",
                    "judge_parse_error": "",
                    "warning": "",
                    "trial_wall_seconds": "0.8",
                    "latency_seconds": "0.4",
                    "response_id": "jr3",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "20",
                    "usage_completion_tokens": "10",
                    "usage_total_tokens": "30",
                    "provider": "{}",
                    "source_prompt_hash": "mno",
                    "judge_prompt_hash": "pqr",
                    "judge_message_preview": "[]",
                    "judge_raw_response_text": "<evaluation></evaluation>",
                },
                {
                    "judge_timestamp": "2026-04-04T10:00:03",
                    "source_run_dir": "source-a",
                    "source_run_name": "source-a",
                    "source_model": "classifier/model",
                    "judge_model": "judge/model",
                    "dataset": "pets",
                    "prompt_type": "features",
                    "config_n": "2",
                    "config_k": "1",
                    "config_q": "9",
                    "run_id": "1",
                    "query_index_within_episode": "0",
                    "predicted_label": "Bombay",
                    "class_options": "[\"Bombay\",\"American Bulldog\"]",
                    "classifier_raw_response_text": "<features>- dark fur</features><response>Bombay</response>",
                    "visual_grounding": "3",
                    "discriminative_support": "3",
                    "inferential_coherence": "3",
                    "clarity": "4",
                    "format_compliance": "4",
                    "overall_score": "3.4000",
                    "judge_parse_error": "",
                    "warning": "",
                    "trial_wall_seconds": "0.8",
                    "latency_seconds": "0.4",
                    "response_id": "jr4",
                    "finish_reason": "stop",
                    "usage_prompt_tokens": "20",
                    "usage_completion_tokens": "10",
                    "usage_total_tokens": "30",
                    "provider": "{}",
                    "source_prompt_hash": "stu",
                    "judge_prompt_hash": "vwx",
                    "judge_message_preview": "[]",
                    "judge_raw_response_text": "<evaluation></evaluation>",
                },
            ]

            with (run_dir / "judge_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(judge_rows[0].keys()))
                writer.writeheader()
                writer.writerows(judge_rows)

            outputs = analyze_judge_run_directory(run_dir)

            self.assertTrue((run_dir / "analysis" / "tables" / "overall_mean_scores_by_prompt.csv").exists())
            self.assertTrue((run_dir / "analysis" / "plots" / "overall_score_by_prompt.png").exists())
            self.assertTrue((run_dir / "analysis" / "stats" / "pairwise_wilcoxon_trial_overall_score.csv").exists())
            self.assertTrue(outputs["report_path"].exists())


if __name__ == "__main__":
    unittest.main()
