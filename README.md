# GeoLLM-Qwen3.5-FineTune

**Domain-adapted language models for mineral exploration and geological knowledge, built on the Qwen 3.5 model family.**

> This fork provides a CPU-first setup for running GeoLLM locally without an NVIDIA GPU or CUDA. CPU inference is slower, but the 0.8B model can run on an older machine such as an Intel Core i5 6th generation.

## CPU Quick Start

### Requirements

- Windows or Linux
- Python 3.10 or 3.11
- RAM:
  - At least 6 GB for the 0.8B model
  - At least 10 GB for the 2B model
  - 16 GB or more for larger models
- At least 10 GB of free disk space for the smallest model
- Internet access to download model files from Hugging Face

> CUDA is not required for CPU inference. Start with the 0.8B model because it is the smallest and most practical option for older systems.

### 1. Clone this branch

```bash
git clone --branch cpu https://github.com/abdulkiqh-oss/GeoLLM-Qwen3.5-FineTune.git
cd GeoLLM-Qwen3.5-FineTune
```

### 2. Create a virtual environment

**Windows:**

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

**Linux:**

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install CPU-only dependencies

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers accelerate safetensors huggingface_hub
```

Do not run `setup_env.sh` or install CUDA, Unsloth, bitsandbytes, or GPU-only packages for CPU inference.

### 4. Run a question

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --question "What geophysical methods can target komatiite-hosted nickel sulphide deposits?"
```

### 5. Run interactive chat

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

For shorter and faster responses:

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --question "Summarize the key gold pathfinder elements." --max_new_tokens 128
```

## CPU-only model loading

The original inference script may use BF16 or automatic GPU/CPU placement. For the widest CPU compatibility, load the model in `float32` and move it explicitly to the CPU:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "AshkanTaghipour/GeoLLM-Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
).to("cpu")
model.eval()

messages = [
    {"role": "system", "content": "You are a specialist geologist."},
    {"role": "user", "content": "What geophysical methods target nickel sulphide deposits?"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )

answer = tokenizer.decode(
    output[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## Recommended models by resource level

| Model | CPU suitable? | Recommended RAM | Expected speed |
|---|---:|---:|---|
| 0.8B | Yes, recommended | 6–8 GB | Slow, but usable |
| 2B | Possible with enough RAM | 10–12 GB | Very slow |
| 4B | Powerful CPU only | 16 GB+ | Very slow |
| 9B | Not recommended | 24 GB+ | Extremely slow |
| 27B | Not practical | 64 GB+ | Generally impractical |

Actual memory and speed depend on the operating system, loaded weights, model cache, and available CPU cores.

## Limitations on an older i5 system

- `train.py` requires an NVIDIA GPU, CUDA, and Unsloth. CPU fine-tuning is not practical.
- Training from scratch or fine-tuning on CPU can take an extremely long time.
- The 9B and 27B models require substantially more memory and processing time.
- An SSD is strongly recommended for faster model loading.
- The first response may take longer because the model and tokenizer are being downloaded and initialized.

## Improving CPU performance

1. Use only the 0.8B model.
2. Limit responses to 128–256 new tokens.
3. Keep prompts and conversations short.
4. Adjust CPU thread usage if the system becomes unresponsive.

**Windows:**

```powershell
set OMP_NUM_THREADS=4
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

**Linux:**

```bash
export OMP_NUM_THREADS=4
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

## Available fine-tuned models

| Model | Hugging Face | VRAM estimate | CPU suitability |
|---|---|---:|---|
| GeoLLM-Qwen3.5-0.8B | [AshkanTaghipour/GeoLLM-Qwen3.5-0.8B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-0.8B) | ~2 GB | Recommended |
| GeoLLM-Qwen3.5-2B | [AshkanTaghipour/GeoLLM-Qwen3.5-2B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-2B) | ~5 GB | High-RAM systems only |
| GeoLLM-Qwen3.5-4B | [AshkanTaghipour/GeoLLM-Qwen3.5-4B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-4B) | ~9 GB | Not recommended |
| GeoLLM-Qwen3.5-9B | [AshkanTaghipour/GeoLLM-Qwen3.5-9B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-9B) | ~19 GB | Not practical |
| GeoLLM-Qwen3.5-27B | [AshkanTaghipour/GeoLLM-Qwen3.5-27B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-27B) | ~55 GB | Not practical |

## Optional NVIDIA GPU usage

CUDA and a modern NVIDIA GPU are optional. They significantly improve inference and are required for practical training. Do not install CUDA packages on a machine that does not have a supported NVIDIA GPU.

## Training and reproduction

Training is designed for NVIDIA GPU systems and depends on CUDA, Unsloth, and bitsandbytes. CPU training is not a practical part of this fork.

```bash
python prepare_data.py --input_dir ./training_splits_v2 --output_dir ./processed_data
python train.py --model_name unsloth/Qwen3.5-0.8B --epochs 5
python benchmark.py --epochs 5
python evaluate.py --model_dir ./finetuned_lora --run_name "cpu-check"
```

## Project structure

```text
├── train.py                 # LoRA training using CUDA/Unsloth
├── evaluate.py              # Model evaluation
├── benchmark.py             # Multi-model benchmarking
├── inference.py             # Standalone and interactive inference
├── prepare_data.py          # Training data preparation
├── model_registry.py        # Qwen 3.5 model registry
├── test_training.py         # Training pipeline tests
├── test_benchmark.py        # Benchmark pipeline tests
├── setup_env.sh             # Original GPU environment setup
├── requirements.txt         # Project dependencies
├── training_splits_v2/      # Raw training data
└── docs/
    ├── BENCHMARK.md
    └── EVALUATION.md
```

## Verify installation

```bash
python -c "import torch, transformers; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
```

For CPU-only installation, `CUDA: False` is the normal and expected result.

## License

The original project is released under Apache-2.0. Fine-tuned model weights also inherit the applicable Qwen model license. Check the license terms before redistribution:

- [Original LICENSE](LICENSE)
- [Qwen 3.5-0.8B license](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE)

Original project by AshkanTaghipour:

- GitHub: https://github.com/AshkanTaghipour
- Hugging Face: https://huggingface.co/AshkanTaghipour
