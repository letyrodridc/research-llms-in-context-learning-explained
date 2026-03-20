from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrouter_mode.analysis import analyze_run_directory
from openrouter_mode.client import OpenRouterClient, extract_message_text, model_supports_images
from openrouter_mode.config import OpenRouterSettings
from openrouter_mode.prompts import pil_image_to_data_url


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
            self.assertTrue((run_dir / "analysis" / "stats" / "pairwise_mcnemar.csv").exists())
            self.assertTrue(outputs["report_path"].exists())


if __name__ == "__main__":
    unittest.main()
