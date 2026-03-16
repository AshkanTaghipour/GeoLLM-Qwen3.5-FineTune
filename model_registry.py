"""
model_registry.py — Registry of Qwen 3.5 dense models for benchmarking.

Defines model configurations (HuggingFace/Unsloth IDs, VRAM requirements,
recommended batch sizes) for all dense Qwen 3.5 models that can be fine-tuned
on a single A100-80GB GPU with bf16 LoRA.

Note: Qwen 3.5 models <=9B are architecturally VLMs (vision-language models)
but are used here in text-only mode via FastLanguageModel. The 27B model is
a dense LLM. All models use the same LoRA target modules and training code.

VRAM estimates are from Unsloth documentation (bf16 LoRA, not QLoRA).
"""

import os

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Target effective batch size for all models (batch_size * grad_accum_steps).
# Kept constant across models for fair comparison.
EFFECTIVE_BATCH_SIZE = 16

MODEL_REGISTRY = {
    "Qwen3.5-0.8B": {
        "model_id": "unsloth/Qwen3.5-0.8B",
        "params_b": 0.8,
        "vram_lora_gb": 3,
        "batch_size": 4,
        "is_vlm": True,
    },
    "Qwen3.5-2B": {
        "model_id": "unsloth/Qwen3.5-2B",
        "params_b": 2.0,
        "vram_lora_gb": 5,
        "batch_size": 4,
        "is_vlm": True,
    },
    "Qwen3.5-4B": {
        "model_id": "unsloth/Qwen3.5-4B",
        "params_b": 4.0,
        "vram_lora_gb": 10,
        "batch_size": 4,
        "is_vlm": True,
    },
    "Qwen3.5-9B": {
        "model_id": "unsloth/Qwen3.5-9B",
        "params_b": 9.0,
        "vram_lora_gb": 22,
        "batch_size": 2,
        "is_vlm": True,
    },
    "Qwen3.5-27B": {
        "model_id": "unsloth/Qwen3.5-27B",
        "params_b": 27.0,
        "vram_lora_gb": 56,
        "batch_size": 1,
        "is_vlm": False,
    },
}

# LoRA target modules — same for all dense Qwen 3.5 models
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_config(model_name):
    """Get the full configuration dict for a registered model.

    Args:
        model_name: Short name like "Qwen3.5-0.8B".

    Returns:
        dict with keys: model_id, params_b, vram_lora_gb, batch_size,
        gradient_accumulation_steps, is_vlm.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    config = dict(MODEL_REGISTRY[model_name])
    config["name"] = model_name
    config["gradient_accumulation_steps"] = (
        EFFECTIVE_BATCH_SIZE // config["batch_size"]
    )
    return config


def get_all_model_names():
    """Return a list of all registered model names, ordered by size."""
    return list(MODEL_REGISTRY.keys())


def get_models_fitting_vram(vram_gb=80):
    """Return model names that fit within the given VRAM budget (bf16 LoRA).

    Args:
        vram_gb: Available GPU VRAM in GB. Default 80 (A100-80GB).

    Returns:
        List of model names sorted by parameter count.
    """
    return [
        name for name, cfg in MODEL_REGISTRY.items()
        if cfg["vram_lora_gb"] <= vram_gb
    ]


def validate_model_names(model_names):
    """Validate that all model names exist in the registry.

    Args:
        model_names: List of model name strings.

    Returns:
        List of validated model names.

    Raises:
        ValueError: If any model name is not in the registry.
    """
    for name in model_names:
        if name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available: {available}"
            )
    return list(model_names)


def resolve_model_path(model_name, local_model_dir="./models"):
    """Resolve a model name to a loadable path.

    Checks if the model exists locally (pre-downloaded). If so, returns the
    local path. Otherwise returns the Unsloth model ID for auto-download.

    Args:
        model_name: Short name like "Qwen3.5-0.8B".
        local_model_dir: Directory where models are cached locally.

    Returns:
        String path — either local directory or Unsloth model ID.
    """
    config = get_model_config(model_name)
    local_path = os.path.join(local_model_dir, model_name)
    if os.path.isdir(local_path):
        return local_path
    return config["model_id"]
