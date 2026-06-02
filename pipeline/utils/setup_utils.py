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
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as F
from torchvision.datasets import Flowers102, OxfordIIITPet, CIFAR10, DTD
from PIL import Image
from torch.utils.data import ConcatDataset

from transformers import AutoProcessor, BitsAndBytesConfig

# These two architectures live in different transformers versions; not every
# install ships both. Import lazily so the module is still usable if only one
# is available (e.g. an environment that has Qwen3-VL but not Gemma4 yet).
try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None  # type: ignore[assignment]
try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
# qwen_vl_utils eliminado: Qwen3-VL integra el procesamiento de imágenes en apply_chat_template

# --- 1. GLOBAL CONFIGURATIONS ---
# Force CUDA kernels to synchronize for easier debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Enable Device-Side Assertions for CUDA
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Configure memory management for the CUDA allocator
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def set_seed(seed=42):
    """Sets the seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. DATASET LOADING ---
class ResizeToWidth:
    """Transformation to resize images to a target width while maintaining aspect ratio."""
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                _, h, w = img.shape
            else:
                h, w = img.shape
        else:
            w, h = img.size

        target_height = int(h * (self.target_width / w))
        return F.resize(img, [target_height, self.target_width], antialias=True)


def load_datasets(data_dir='./data'):
    """Loads benchmark datasets (Flowers102, OxfordPets, CIFAR10, DTD). Downloads only if necessary."""
    os.makedirs(data_dir, exist_ok=True)
    datasets = {}
    transform = T.Compose([
        T.ToTensor(),
        ResizeToWidth(target_width=224),
    ])

    print(f"Loading Flower102 into {data_dir}...")
    # torchvision datasets download only if not present when download=True
    train_set = Flowers102(root=data_dir, split='train', download=True, transform=transform)
    val_set   = Flowers102(root=data_dir, split='val', download=True, transform=transform)
    flowers_full = ConcatDataset([train_set, val_set])
    datasets['flowers'] = flowers_full
    # Flowers102 in torchvision does not expose a .classes attribute; use the
    # standard Oxford 102 Flowers category names (0-indexed, matching _labels).
    datasets['flowers_classes'] = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
        "sword lily", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris",
        "globe-thistle", "purple coneflower", "peruvian lily", "balloon flower",
        "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
        "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
        "stemless gentian", "artichoke", "sweet william", "carnation",
        "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
        "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup", "daisy",
        "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
        "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
        "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
        "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
        "spring crocus", "iris", "windflower", "tree poppy", "gazania", "azalea",
        "water lily", "rose", "thorn apple", "morning glory", "passion flower",
        "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus",
        "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
        "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
        "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia",
        "bromelia", "blanket flower", "trumpet creeper", "blackberry lily",
    ]

    print(f"Loading OxfordPets into {data_dir}...")
    datasets['pets'] = OxfordIIITPet(root=data_dir, split='trainval', download=True, transform=transform)
    datasets['pets_classes'] = datasets['pets'].classes

    print(f"Loading CIFAR-10 into {data_dir}...")
    datasets['cifar10'] = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    datasets['cifar10_classes'] = datasets['cifar10'].classes

    print(f"Loading DTD into {data_dir}...")
    datasets['dtd'] = DTD(root=data_dir, split='train', download=True, transform=transform)
    datasets['dtd_classes'] = datasets['dtd'].classes

    print("All datasets loaded successfully.")
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
    "gemma4":                  "/gpfs/projects/ugr92/ICL/hf_cache/gemma-4-26B-A4B-it",
    "qwen3-vl":                "/gpfs/projects/ugr92/ICL/hf_cache/Qwen3-VL-8B-Instruct",
    # Local LLM-as-a-judge model used by pipeline/evaluation/local_judge.py.
    "qwen3-vl-32b-thinking":   "/gpfs/projects/ugr92/ICL/hf_cache/Qwen3-VL-32B-Thinking",
}

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - allocated
        print(f"GPU {i}: Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Free={free:.2f}GB / {total:.2f}GB")


def load_model_globally(model_name, *, quantization="auto"):
    """Loads a specified model and its processor.

    Args:
        model_name: either a key registered in :data:`MODEL_IDS`
            (e.g. ``"qwen3-vl"``, ``"qwen3-vl-32b-thinking"``, ``"gemma4"``)
            **or** a direct model path / Hugging Face hub repo id
            (e.g. ``"Qwen/Qwen3-VL-2B-Instruct"`` or ``"/gpfs/.../my-model"``).
            Architecture is dispatched on substring match against the lowered
            name (``"gemma4"`` / ``"qwen3-vl"``), so direct IDs need to keep
            those substrings in them.
        quantization: ``"auto"`` (default mapping per model), ``"bf16"`` (no
            quantization), or ``"nf4"`` (4-bit NF4 via bitsandbytes). The
            32B-Thinking judge defaults to BF16; the 8B inference model
            defaults to NF4.
    """
    torch.cuda.empty_cache()
    gc.collect()

    key = model_name.lower()
    model_id = MODEL_IDS.get(key, model_name)
    # `key` drives the architecture branch below; if the caller passed a raw
    # HF id like "Qwen/Qwen3-VL-2B-Instruct" we still want the qwen3-vl branch.

    print(f"Loading model: {model_id}...")

    if "gemma4" in key:
        if AutoModelForImageTextToText is None:
            raise ImportError(
                "AutoModelForImageTextToText is not available in this transformers "
                "install. Upgrade transformers to a version that ships gemma4 support."
            )
        # Gemma4 always uses NF4 in this repo regardless of `quantization`.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    elif "qwen3-vl" in key:
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available in this transformers "
                "install. Upgrade transformers to a version that ships Qwen3-VL support."
            )
        # Default policy: 8B inference -> NF4, 32B-Thinking judge -> BF16.
        if quantization == "auto":
            # NF4 for all Qwen3-VL variants: the 32B model in BF16 fills ~64 GB of
            # an 80 GB H100, leaving only ~16 GB for KV-cache and causing ~3 tok/s
            # generation. NF4 reduces weights to ~18 GB and restores normal throughput
            # (~30-50 tok/s) with negligible quality loss on a 1-5 scoring rubric.
            quantization = "nf4"

        if quantization == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        elif quantization == "bf16":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError(
                f"Unsupported quantization mode for {model_name}: {quantization}. "
                "Use 'auto', 'bf16', or 'nf4'."
            )

    else:
        raise ValueError(
            f"Could not infer architecture from model name `{model_name}`. "
            f"Expected the name to contain 'gemma4' or 'qwen3-vl', or to be a "
            f"key in MODEL_IDS ({list(MODEL_IDS.keys())})."
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
    #random.shuffle(support_idx)
    
    # 2. Build the lists using the shuffled indices
    indices = support_idx + [data['query_indices'][index]]
    shots = [dataset[i] for i in support_idx]
    query = dataset[data['query_indices'][index]]
    
    return indices, shots, query, class_names

def extract_response(text):
    """Extracts the content within <response> tags."""
    match = re.search(r'<response>(.*?)</response>', text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def decode_data_url(data_url):
    """Decodes a base64 data URL into a PIL Image."""
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None

def run_icl_inference(model, processor, model_name, messages, content_parts=None, temperature=0.2, max_new_tokens=1024):
    """Runs In-Context Learning inference handling Qwen vs Gemma differences."""
    
    # Pre-process messages to convert data URLs to PIL images for local consumption
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image_url" and "url" in part.get("image_url", {}):
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        part["type"] = "image"
                        part["image"] = decode_data_url(url)
     
    # Gemma4 y Qwen3-VL usan el mismo path unificado:
    # apply_chat_template con tokenize=True maneja imágenes y texto en un solo paso.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    inputs.pop("token_type_ids", None)

    # with torch.no_grad():
    #     generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
    
    if os.environ.get("DEBUG_VERBOSE") == "1":
        try:
            full_decoded = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            full_decoded = ["<decoding_failed>"]
        print("DEBUG input_ids lengths:", [len(x) for x in inputs.input_ids])
        print("DEBUG full_generated:", full_decoded)
           
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