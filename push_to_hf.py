"""
push_to_hf.py — Upload fine-tuned models and dataset to HuggingFace.

This script creates HuggingFace repos and uploads:
  - 5 merged bf16 models (one repo each)
  - 1 dataset repo

Usage:
    python push_to_hf.py --all              # Upload everything
    python push_to_hf.py --models           # Upload models only
    python push_to_hf.py --dataset          # Upload dataset only
    python push_to_hf.py --model Qwen3.5-0.8B  # Upload one model
"""

import argparse
import json
import os

HF_USERNAME = "AshkanTaghipour"
MODEL_PREFIX = "GeoLLM-Qwen3.5"
DATASET_REPO = f"{HF_USERNAME}/mineral-exploration-geology-qa"
GITHUB_REPO = "https://github.com/AshkanTaghipour/GeoLLM-Qwen3.5-FineTune"

# Map model short names to their merged model directories
MODEL_DIRS = {
    "Qwen3.5-0.8B": "benchmark_results/benchmark-20260315-203230/merged_models/Qwen3.5-0.8B",
    "Qwen3.5-2B": "benchmark_results/benchmark-20260315-203230/merged_models/Qwen3.5-2B",
    "Qwen3.5-4B": "benchmark_results/benchmark-20260315-203230/merged_models/Qwen3.5-4B",
    "Qwen3.5-9B": "benchmark_results/benchmark-20260315-203230/merged_models/Qwen3.5-9B",
    "Qwen3.5-27B": "benchmark_results/benchmark-20260316-025347/merged_models/Qwen3.5-27B",
}

# Benchmark scores for model cards
SCORES = {
    "Qwen3.5-0.8B": {"params": "0.8B", "base": 0.345, "ft": 0.351, "loss": 1.863,
                       "qa_rougeL": 0.1697, "qa_bert": 0.8447, "cot_rougeL": 0.2028,
                       "halluc": "40.0%", "vram_train": "3 GB", "vram_infer": "~2 GB"},
    "Qwen3.5-2B":   {"params": "2.0B", "base": 0.355, "ft": 0.343, "loss": 1.576,
                       "qa_rougeL": 0.1869, "qa_bert": 0.8475, "cot_rougeL": 0.1972,
                       "halluc": "26.7%", "vram_train": "5 GB", "vram_infer": "~5 GB"},
    "Qwen3.5-4B":   {"params": "4.0B", "base": 0.341, "ft": 0.353, "loss": 1.316,
                       "qa_rougeL": 0.1931, "qa_bert": 0.8533, "cot_rougeL": 0.1953,
                       "halluc": "33.3%", "vram_train": "10 GB", "vram_infer": "~9 GB"},
    "Qwen3.5-9B":   {"params": "9.0B", "base": 0.351, "ft": 0.343, "loss": 1.172,
                       "qa_rougeL": 0.1947, "qa_bert": 0.8525, "cot_rougeL": 0.2064,
                       "halluc": "20.0%", "vram_train": "22 GB", "vram_infer": "~19 GB"},
    "Qwen3.5-27B":  {"params": "27.0B", "base": 0.343, "ft": 0.361, "loss": 1.005,
                       "qa_rougeL": 0.1939, "qa_bert": 0.8526, "cot_rougeL": 0.2335,
                       "halluc": "33.3%", "vram_train": "56 GB", "vram_infer": "~55 GB"},
}


def get_hf_token():
    """Read HF token from credentials file."""
    creds_path = os.path.join(os.path.dirname(__file__), "credentials.txt")
    with open(creds_path) as f:
        for line in f:
            if "hf_tok" in line.lower():
                return line.split(":", 1)[1].strip()
    raise RuntimeError("HF token not found in credentials.txt")


def make_model_card(model_name):
    """Generate a HuggingFace model card (README.md) for a model."""
    s = SCORES[model_name]
    hf_repo = f"{HF_USERNAME}/{MODEL_PREFIX}-{model_name.split('-')[-1]}"
    base_model = f"Qwen/{model_name}"

    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - geology
  - mineral-exploration
  - qwen3.5
  - lora
  - fine-tuned
  - geoscience
datasets:
  - {DATASET_REPO}
language:
  - en
pipeline_tag: text-generation
---

# {MODEL_PREFIX}-{model_name.split('-')[-1]}

Fine-tuned **{model_name}** for mineral exploration geology, targeting Western Australian and Queensland geological domains.

This model is part of the [GeoLLM-Qwen3.5-FineTune]({GITHUB_REPO}) benchmark. All five model sizes (0.8B--27B) were trained with identical hyperparameters on the same dataset for fair comparison.

## Performance

| Metric | Base | Fine-tuned |
|:-------|-----:|-----------:|
| Overall weighted score | {s['base']} | **{s['ft']}** |
| QA ROUGE-L | -- | {s['qa_rougeL']} |
| QA BERTScore | -- | {s['qa_bert']} |
| CoT ROUGE-L | -- | {s['cot_rougeL']} |
| Hallucination pass rate | -- | {s['halluc']} |
| Training loss | -- | {s['loss']} |

See the [full benchmark comparison]({GITHUB_REPO}/blob/main/docs/BENCHMARK.md) for all models.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{hf_repo}", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{hf_repo}")

messages = [
    {{"role": "system", "content": "You are a specialist geologist with expertise in Western Australian and Queensland mineral exploration."}},
    {{"role": "user", "content": "What geophysical methods would you recommend for targeting komatiite-hosted nickel sulphide deposits in the Eastern Goldfields?"}},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Training Details

- **Method:** bf16 LoRA (r=16, alpha=16) via Unsloth + SFTTrainer
- **Epochs:** 5
- **Optimizer:** adamw_8bit, cosine LR schedule (2e-4, 10% warmup)
- **Dataset:** [mineral-exploration-geology-qa]({DATASET_REPO}) (479 train, 26 test examples)
- **Hardware:** NVIDIA A100-80GB
- **VRAM (training):** {s['vram_train']}
- **VRAM (inference):** {s['vram_infer']}

## Author

**Ashkan Taghipour** -- [GitHub]({GITHUB_REPO}) | [HuggingFace](https://huggingface.co/{HF_USERNAME})
"""


def make_dataset_card():
    """Generate a HuggingFace dataset card."""
    return f"""---
license: apache-2.0
tags:
  - geology
  - mineral-exploration
  - geoscience
  - question-answering
  - chain-of-thought
language:
  - en
size_categories:
  - n<1K
---

# Mineral Exploration Geology QA Dataset

Expert-curated question-answer pairs for mineral exploration geology, targeting Western Australian and Queensland geological domains.

Used for fine-tuning the [GeoLLM-Qwen3.5-FineTune]({GITHUB_REPO}) model family.

## Dataset Summary

| Split | QA Pairs | CoT Pairs | Total |
|:------|:---------|:----------|:------|
| Train | 352 | 127 | 479 |
| Test | 19 | 7 | 26 |
| Hallucination traps | -- | -- | 15 |

## Format

**QA pairs** (`qa_pairs.jsonl`):
```json
{{"question": "What are the key pathfinder elements for orogenic gold?", "answer": "The primary pathfinders are As, Sb, W, Bi, and Te..."}}
```

**CoT pairs** (`cot_pairs.jsonl`):
```json
{{"question": "...", "reasoning": "Step-by-step geological reasoning...", "answer": "Final answer..."}}
```

**Hallucination traps** (`hallucination_traps.jsonl`):
```json
{{"id": "trap_001", "category": "fictional_deposit", "question": "...", "trap_detail": "...", "acceptable_responses": ["..."]}}
```

## Topics Covered

- Geological interpretation and structural geology
- Exploration targeting and tenement evaluation
- Deposit models: orogenic gold, VMS, komatiite Ni, IOCG, channel iron, REE
- Geochemistry: pathfinder elements, lithogeochemistry, regolith sampling
- Geophysics: magnetics, gravity, EM, IP, radiometrics
- Drilling strategies: RC, diamond, aircore
- Western Australian and Queensland regulatory frameworks

## Usage

```python
import json

with open("qa_pairs.jsonl") as f:
    qa_data = [json.loads(line) for line in f if line.strip()]

print(f"{{len(qa_data)}} QA pairs loaded")
print(qa_data[0]["question"])
```

## Author

**Ashkan Taghipour** -- [GitHub]({GITHUB_REPO}) | [HuggingFace](https://huggingface.co/{HF_USERNAME})
"""


def upload_model(model_name, token):
    """Upload a single merged model to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    size_label = model_name.split("-")[-1]
    repo_id = f"{HF_USERNAME}/{MODEL_PREFIX}-{size_label}"
    local_dir = MODEL_DIRS[model_name]

    if not os.path.isdir(local_dir):
        print(f"[skip] {model_name}: directory not found at {local_dir}")
        return False

    print(f"\n[upload] {model_name} -> {repo_id}")
    print(f"  Local: {local_dir}")

    # Create repo (no-op if exists)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Write model card
    card_path = os.path.join(local_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(make_model_card(model_name))

    # Upload all files
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        repo_type="model",
        commit_message=f"Upload {model_name} fine-tuned for mineral exploration geology",
    )
    print(f"  Done: https://huggingface.co/{repo_id}")
    return True


def upload_dataset(token):
    """Upload the dataset to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    data_dir = "training_splits_v2"

    if not os.path.isdir(data_dir):
        print(f"[skip] Dataset directory not found at {data_dir}")
        return False

    print(f"\n[upload] Dataset -> {DATASET_REPO}")

    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

    # Write dataset card
    card_path = os.path.join(data_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(make_dataset_card())

    api.upload_folder(
        repo_id=DATASET_REPO,
        folder_path=data_dir,
        repo_type="dataset",
        commit_message="Upload mineral exploration geology QA dataset",
    )
    print(f"  Done: https://huggingface.co/datasets/{DATASET_REPO}")
    return True


def parse_upload_args():
    p = argparse.ArgumentParser(description="Upload models/dataset to HuggingFace.")
    p.add_argument("--all", action="store_true", help="Upload everything.")
    p.add_argument("--models", action="store_true", help="Upload all models.")
    p.add_argument("--dataset", action="store_true", help="Upload dataset.")
    p.add_argument("--model", help="Upload a specific model (e.g., Qwen3.5-0.8B).")
    return p.parse_args()


def main():
    args = parse_upload_args()
    token = get_hf_token()

    if not (args.all or args.models or args.dataset or args.model):
        print("Specify --all, --models, --dataset, or --model <name>. See --help.")
        return

    if args.all or args.dataset:
        upload_dataset(token)

    if args.all or args.models:
        for name in MODEL_DIRS:
            upload_model(name, token)
    elif args.model:
        if args.model not in MODEL_DIRS:
            print(f"Unknown model '{args.model}'. Available: {list(MODEL_DIRS.keys())}")
            return
        upload_model(args.model, token)


if __name__ == "__main__":
    main()
