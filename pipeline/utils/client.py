from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import time

import requests

from ..experiments.config import OpenRouterSettings


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns a non-recoverable error."""


@dataclass
class OpenRouterResponse:
    text: str
    raw_json: Dict[str, Any]
    latency_seconds: float
    status_code: int
    request_id: Optional[str]
    provider: Any
    usage: Dict[str, Any]
    finish_reason: Optional[str]


def extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(part.get("text", ""))
            elif "text" in part:
                text_parts.append(str(part.get("text", "")))
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)


class OpenRouterClient:
    def __init__(
        self,
        settings: OpenRouterSettings,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.settings.app_name,
        }
        if self.settings.site_url:
            headers["HTTP-Referer"] = self.settings.site_url
        return headers

    def create_chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> OpenRouterResponse:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if generation_params:
            payload.update(generation_params)

        last_error: Optional[str] = None

        for attempt in range(1, self.settings.max_retries + 1):
            start = time.perf_counter()
            response = self.session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            latency_seconds = time.perf_counter() - start

            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.settings.max_retries:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    sleep_seconds = float(retry_after)
                else:
                    sleep_seconds = min(2 ** (attempt - 1), 8)
                time.sleep(sleep_seconds)
                continue

            if not response.ok:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {"error": response.text}
                last_error = json.dumps(error_payload, ensure_ascii=False)
                raise OpenRouterError(
                    f"OpenRouter request failed with status {response.status_code}: {last_error}"
                )

            data = response.json()
            choice = (data.get("choices") or [{}])[0]

            return OpenRouterResponse(
                text=extract_message_text(data),
                raw_json=data,
                latency_seconds=latency_seconds,
                status_code=response.status_code,
                request_id=data.get("id") or response.headers.get("x-request-id"),
                provider=data.get("provider"),
                usage=data.get("usage") or {},
                finish_reason=choice.get("finish_reason"),
            )

        raise OpenRouterError(last_error or "OpenRouter request failed after retries.")

    def fetch_model_metadata(self) -> Optional[Dict[str, Any]]:
        response = self.session.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=self._headers(),
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        for model_info in data:
            if model_info.get("id") == self.settings.model:
                return model_info
        return None


def model_supports_images(model_info: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not model_info:
        return None
    architecture = model_info.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    return "image" in input_modalities
