import os
import gc
import torch
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .setup_utils import run_icl_inference
from .client import OpenRouterClient, OpenRouterError
from ..experiments.config import build_openrouter_settings

class InferenceEngine:
    """Unified engine for both local and OpenRouter inference."""
    
    def __init__(self, mode: str, model_name: str, env_file: Optional[str] = None):
        self.mode = mode
        self.model_name = model_name
        self.local_model = None
        self.local_processor = None
        self.or_client = None
        
        if mode == "local":
            from .setup_utils import load_model_globally
            self.local_model, self.local_processor = load_model_globally(model_name)
        else:
            # OpenRouter mode
            env_path = Path(env_file).resolve() if env_file else Path(".env").resolve()
            settings = build_openrouter_settings(env_path, cli_model=model_name)
            self.or_client = OpenRouterClient(settings)

    def infer(self, messages: List[Dict[str, Any]], max_tokens: int, temperature: float = 0.0) -> str:
        """
        Executes inference based on the selected mode (local or OpenRouter).
        Includes automatic fallback for OpenRouter providers that do not support system prompts.
        """
        if self.mode == "local":
            # Direct inference using locally loaded model
            return run_icl_inference(
                self.local_model, 
                self.local_processor, 
                self.model_name, 
                messages, 
                temperature=temperature, 
                max_new_tokens=max_tokens
            )
        else:
            # Inference via OpenRouter API
            try:
                response = self.or_client.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.text
            except Exception as e:
                # Fallback: if 'system prompt' or 'developer instruction' is rejected by the provider,
                # retry by folding the system message into the first user message.
                if "developer instruction" in str(e).lower() or "system prompt" in str(e).lower():
                    from ..experiments.experiment_config import flatten_system_prompt_into_first_user_message
                    new_messages = flatten_system_prompt_into_first_user_message(messages)
                    response = self.or_client.create_chat_completion(
                        messages=new_messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.text
                raise e
