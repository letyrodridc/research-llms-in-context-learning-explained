import os
import sys
import gc
import random
import numpy as np
import itertools
import traceback
import re

import torch
import torchvision
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.datasets import Flowers102, OxfordIIITPet, CIFAR10, DTD
from PIL import Image

from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Gemma3ForConditionalGeneration,
    Qwen2VLForConditionalGeneration
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass  # If you only install transformers, ignore this error

# --- 1. GLOBAL CONFIGURATIONS ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. DATASET LOADING ---
class ResizeToWidth:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size

        target_height = int(h * (self.target_width / w))
        return F.resize(img, [target_height, self.target_width], antialias=True)


def load_datasets(data_dir='./data'):
    """Downloads and loads all four datasets as defined in the notebook."""
    datasets = {}
    transform = T.Compose([
        T.ToTensor(),
        # ResizeToWidth(target_width=96),
        ResizeToWidth(target_width=224),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading Flower102...")
    datasets['flowers'] = Flowers102(root=data_dir, split='train', download=True, transform=transform)

    print("Loading OxfordPets...")
    datasets['pets'] = OxfordIIITPet(root=data_dir, split='trainval', download=True, transform=transform)
    datasets['pets_classes'] = datasets['pets'].classes

    print("Loading CIFAR-10...")
    datasets['cifar10'] = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    datasets['cifar10_classes'] = datasets['cifar10'].classes

    print("Loading DTD...")
    datasets['dtd'] = DTD(root=data_dir, split='train', download=True, transform=transform)
    datasets['dtd_classes'] = datasets['dtd'].classes

    print("All datasets loaded.")
    return datasets


def build_class_index_map(dataset):
    """Creates a dictionary mapping class_id -> list of indices."""
    from collections import defaultdict
    class_indices = defaultdict(list)

    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if hasattr(targets, 'tolist'):
            targets = targets.tolist()
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
    else:
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_indices[label].append(idx)

    return class_indices

# --- 3. MODEL LOADING ---
MODEL_IDS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "gemma3": "google/gemma-3-12b-it",
    "bakllava": "llava-hf/bakLlava-v1-hf",
    "qwen-vl": "Qwen/Qwen2-VL-7B-Instruct",
}

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - allocated
        print(f"GPU {i}: Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Free={free:.2f}GB / {total:.2f}GB")


def load_model_globally(model_name):
    """Loads a specified model and its processor."""
    torch.cuda.empty_cache()
    gc.collect()
    
    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_IDS.keys())}")

    print(f"Loading model: {model_id}...")

    if "gemma" in model_name.lower():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
    elif "qwen-vl" in model_name.lower():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model {model_id} loaded successfully.")
    print_gpu_memory()
    
    return model, processor


def select_few_shot_images_with_data_fixed(data, index, dataset, class_names):
    """
    Adjusted version so that it does not depend on global variables; 
    it receives the dataset and class_names as parameters.
    """
    # 1. Extract and explicitly SHUFFLE the support indices
    support_idx = list(data['support_indices'])
    random.shuffle(support_idx)
    
    # 2. Build the lists using the shuffled indices
    indices = support_idx + [data['query_indices'][index]]
    shots = [dataset[i] for i in support_idx]
    query = dataset[data['query_indices'][index]]
    
    return indices, shots, query, class_names

def run_icl_inference(model, processor, model_name, messages, content_parts=None, temperature=0.2, max_new_tokens=1024):
    """Runs In-Context Learning inference handling Qwen vs Gemma differences."""
     
    if "qwen-vl" in model_name.lower():
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if content_parts:
            all_images_pil = [part["image"] for part in content_parts if part["type"] == "image"]
        else:
            content_parts_flat = [part["content"] for part in messages if type(part["content"]) == list]
            content_parts_flat = list(itertools.chain.from_iterable(content_parts_flat))
            all_images_pil = [part["image"] for part in content_parts_flat if part["type"] == "image"]

        inputs = processor(
            text=[text_prompt],
            images=all_images_pil,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)
    
    # with torch.no_grad():
    #     generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    
    return output_text[0]