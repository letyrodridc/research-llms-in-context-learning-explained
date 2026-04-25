import os
import csv
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..utils.inference_engine import InferenceEngine
from .reconstruction import reconstruct_classifier_messages
from ..utils.setup_utils import load_datasets

SCORE_FIELDS = [
    "visual_grounding",
    "discriminative_support",
    "inferential_coherence",
    "clarity",
    "format_compliance",
]

class JudgeRunner:
    """
    Core class for executing LLM-as-a-judge evaluations.
    Reconstructs the original classifier context and uses a judge model to score explanations.
    """
    def __init__(self, mode: str, model_name: str, judge_library_path: str, run_dir: str):
        self.engine = InferenceEngine(mode, model_name)
        self.judge_library = self._load_judge_library(judge_library_path)
        self.run_dir = Path(run_dir)
        self.datasets_dict = load_datasets()

    def _load_judge_library(self, path):
        """Loads the judge prompt library from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_scores(self, text: str) -> Dict[str, str]:
        """Parses the judge's XML response to extract scores for each dimension."""
        scores = {}
        for field in SCORE_FIELDS:
            # More robust regex: handle optional decimals (e.g., 4.0), 
            # varied whitespace, and case-insensitive tags.
            pattern = fr"<{field}>\s*([1-5](?:\.\d+)?)\s*</{field}>"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                val = match.group(1)
                # Convert to int string if it's something like 4.0
                scores[field] = str(int(float(val)))
            else:
                scores[field] = "N/A"
        return scores

    def build_judge_messages(self, row: Dict[str, str], reconstructed_messages: List[Dict[str, Any]]):
        """
        Constructs the evaluation prompt for the judge model.
        Includes the classifier's predicted label, the candidate output, and the target image.
        """
        prompt_type = row['prompt_type']
        spec = self.judge_library['prompt_types'][prompt_type]
        
        system_prompt = self.judge_library['shared_system_prompt'].replace(
            "{CONDITION_DESCRIPTION}", spec['condition_description']
        )
        
        user_text = (
            "Evaluate the candidate classifier output using the exact classifier context below.\n"
            f"Explanation condition: {prompt_type}\n"
            f"Candidate class labels: {row.get('class_options', '[]')}\n"
            f"Classifier predicted label: {row.get('predicted_label', '<missing>')}\n"
            "--- CLASSIFIER CONVERSATION ---\n"
        )
        
        # Extract the query image from the reconstructed classifier conversation
        target_image_url = None
        for msg in reversed(reconstructed_messages):
            if isinstance(msg['content'], list):
                for part in msg['content']:
                    if part['type'] == 'image_url':
                        target_image_url = part['image_url']['url']
                        break
            if target_image_url: break

        judge_content = [{"type": "text", "text": user_text}]
        if target_image_url:
            # Provide the target image to the judge for visual grounding assessment
            judge_content.append({"type": "image_url", "image_url": {"url": target_image_url}})
            
        judge_content.append({
            "type": "text", 
            "text": f"Classifier Output to evaluate:\n{row.get('raw_response_text', '')}"
        })

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": judge_content}
        ]

    def run(self, limit: Optional[int] = None):
        trial_results_path = self.run_dir / "trial_results.csv"
        if not trial_results_path.exists():
            raise FileNotFoundError(f"Missing trial_results.csv in {self.run_dir}")

        output_dir = self.run_dir / "judge_outputs"
        output_dir.mkdir(exist_ok=True)
        csv_output = output_dir / f"judge_results_{self.engine.model_name.replace('/', '_')}.csv"
        
        with open(trial_results_path, 'r', encoding='utf-8') as f:
            trials = list(csv.DictReader(f))

        if limit:
            trials = trials[:limit]

        fieldnames = ["dataset", "prompt_type", "run_id", "query_index"] + SCORE_FIELDS + ["raw_judge_output"]
        
        with open(csv_output, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(trials):
                # Only judge trials with explanations
                if row['prompt_type'] == 'classification': continue
                if row['error']: continue

                print(f"[*] Judging trial {i+1}/{len(trials)}: {row['dataset']} | {row['prompt_type']}")
                
                dataset = self.datasets_dict[row['dataset']]
                class_names = self.datasets_dict.get(f"{row['dataset']}_classes")
                
                reconstructed = reconstruct_classifier_messages(row, dataset, class_names)
                messages = self.build_judge_messages(row, reconstructed.classifier_messages)
                
                prompt_spec = self.judge_library['prompt_types'][row['prompt_type']]
                judge_output = self.engine.infer(messages, max_tokens=prompt_spec['max_tokens'])
                
                scores = self._extract_scores(judge_output)
                
                result_row = {
                    "dataset": row['dataset'],
                    "prompt_type": row['prompt_type'],
                    "run_id": row['run_id'],
                    "query_index": row['query_index_within_episode'],
                    **scores,
                    "raw_judge_output": judge_output
                }
                writer.writerow(result_row)
                f_out.flush()

        print(f"[+] Judging finished. Results saved to {csv_output}")
