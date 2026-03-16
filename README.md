# GeoLLM-Qwen3.5-FineTune

**LoRA fine-tuning and benchmarking of the Qwen 3.5 model family (0.8B -- 27B) for mineral exploration geology, targeting the Western Australian geological domain.**

Fine-tuned models are available on HuggingFace and can be used for geological interpretation, exploration planning, deposit model analysis, geochemistry, geophysics, and drilling strategy questions.

---

## Key Results

All models were trained for **5 epochs** with **bf16 LoRA** (r=16, alpha=16) on an A100-80GB GPU using the same dataset of 479 geology QA and chain-of-thought examples.

| Model | Params | Base Score | Fine-tuned Score | Train Loss | Wall Time |
|:------|-------:|-----------:|-----------------:|-----------:|----------:|
| [Qwen3.5-0.8B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-0.8B) | 0.8B | 0.345 | 0.351 | 1.863 | 15 min |
| [Qwen3.5-2B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-2B) | 2.0B | 0.355 | 0.343 | 1.576 | 15 min |
| [Qwen3.5-4B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-4B) | 4.0B | 0.341 | 0.353 | 1.316 | 22 min |
| [Qwen3.5-9B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-9B) | 9.0B | 0.351 | 0.343 | 1.172 | 37 min |
| [Qwen3.5-27B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-27B) | 27.0B | 0.343 | **0.361** | 1.005 | 133 min |

> **Overall weighted score** = 35% QA ROUGE-L + 25% QA BERTScore + 20% CoT ROUGE-L + 10% think-tag rate + 10% hallucination pass rate.
> See [Benchmark Details](docs/BENCHMARK.md) for full per-metric tables and [Evaluation Methodology](docs/EVALUATION.md) for metric definitions.

### Highlights

- **27B achieves the best fine-tuned score (0.361)**, benefiting most from domain adaptation
- **Training loss scales inversely with model size** (1.86 for 0.8B down to 1.00 for 27B), indicating larger models fit the geology domain more efficiently
- **ROUGE and BERTScore improve across all models** after fine-tuning, with 4B and 27B showing the most consistent gains
- **Hallucination pass rate drops after fine-tuning** for most models, a known trade-off where domain-adapted models become more assertive but less cautious on fabricated premises
- **0.8B offers the best cost-efficiency**: competitive scores in 15 minutes of training, suitable for deployment on consumer hardware

---

## Models

All fine-tuned models are available on HuggingFace as merged bf16 checkpoints (ready for direct inference, no adapter loading required):

| Model | HuggingFace | VRAM (inference) |
|:------|:------------|:-----------------|
| GeoLLM-Qwen3.5-0.8B | [AshkanTaghipour/GeoLLM-Qwen3.5-0.8B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-0.8B) | ~2 GB |
| GeoLLM-Qwen3.5-2B | [AshkanTaghipour/GeoLLM-Qwen3.5-2B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-2B) | ~5 GB |
| GeoLLM-Qwen3.5-4B | [AshkanTaghipour/GeoLLM-Qwen3.5-4B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-4B) | ~9 GB |
| GeoLLM-Qwen3.5-9B | [AshkanTaghipour/GeoLLM-Qwen3.5-9B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-9B) | ~19 GB |
| GeoLLM-Qwen3.5-27B | [AshkanTaghipour/GeoLLM-Qwen3.5-27B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-27B) | ~55 GB |

---

## Quick Start

### Inference (using a fine-tuned model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "AshkanTaghipour/GeoLLM-Qwen3.5-4B"  # or any model above
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16", device_map="auto")

messages = [
    {"role": "system", "content": "You are a specialist geologist with expertise in Western Australian mineral exploration."},
    {"role": "user", "content": "What geophysical methods would you recommend for targeting komatiite-hosted nickel sulphide deposits in the Eastern Goldfields?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

A standalone inference script is also provided:

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-4B --question "What are the key pathfinder elements for orogenic gold in the Yilgarn Craton?"
```

### Training (reproduce from scratch)

```bash
# 1. Set up environment
bash setup_env.sh

# 2. Prepare data
python prepare_data.py --input_dir ./training_splits_v2 --output_dir ./processed_data

# 3. Train a single model
python train.py --model_name unsloth/Qwen3.5-0.8B --epochs 5

# 4. Run the full benchmark (all 5 models)
python benchmark.py --epochs 5
```

### Evaluation

```bash
# Evaluate a fine-tuned model
python evaluate.py --model_dir ./finetuned_lora --run_name "my-eval"
```

---

## Dataset

The training dataset was constructed from ~300 recent mineral exploration reports sourced from [WAMEX](https://www.dmp.wa.gov.au/WAMEX-Minerals-Exploration-1476.aspx) (Western Australia Mineral Exploration Index). Reports were OCR-processed, then an LLM generated expert-style QA and chain-of-thought pairs that were human-reviewed for geological accuracy. See the [dataset card](https://huggingface.co/datasets/AshkanTaghipour/mineral-exploration-geology-qa) for full methodology.

| Split | QA Pairs | CoT Pairs | Total |
|:------|:---------|:----------|:------|
| Train | 352 | 127 | 479 |
| Test | 19 | 7 | 26 |
| Hallucination traps | -- | -- | 15 |

Available on HuggingFace: [AshkanTaghipour/mineral-exploration-geology-qa](https://huggingface.co/datasets/AshkanTaghipour/mineral-exploration-geology-qa)

Topics include: geological interpretation, exploration targeting, deposit models (orogenic gold, VMS, komatiite Ni, IOCG, channel iron), geochemistry, geophysics (magnetics, gravity, EM, IP), drilling strategies, and regulatory frameworks.

---

## Project Structure

```
├── train.py                 # LoRA fine-tuning (Unsloth + SFTTrainer)
├── evaluate.py              # Evaluation (ROUGE, BERTScore, hallucination traps)
├── benchmark.py             # Multi-model benchmark orchestrator
├── inference.py             # Standalone inference script
├── prepare_data.py          # Data transformation pipeline
├── model_registry.py        # Qwen 3.5 model family registry
├── test_training.py         # Training pipeline tests
├── test_benchmark.py        # Benchmark pipeline tests
├── setup_env.sh             # Environment setup
├── requirements.txt         # Python dependencies
├── training_splits_v2/      # Raw training data (JSONL)
└── docs/
    ├── BENCHMARK.md          # Detailed benchmark results and analysis
    └── EVALUATION.md         # Evaluation methodology
```

---

## Documentation

- **[Benchmark Results and Analysis](docs/BENCHMARK.md)** -- Full per-metric tables, training statistics, and key findings across all five model sizes
- **[Evaluation Methodology](docs/EVALUATION.md)** -- Metric definitions, hallucination trap design, and the weighted scoring formula

---

## Acknowledgments

This project builds on the work and resources collected in the [mineral-exploration-machine-learning](https://github.com/RichardScottOZ/mineral-exploration-machine-learning) repository by Richard Scott, which provides a comprehensive catalog of machine learning applications in mineral exploration and geoscience.

---

## Author

**Ashkan Taghipour**
- GitHub: [AshkanTaghipour](https://github.com/AshkanTaghipour)
- HuggingFace: [AshkanTaghipour](https://huggingface.co/AshkanTaghipour)

---

## License

This project is released under the [Apache 2.0 License](LICENSE). The fine-tuned model weights inherit the [Qwen License](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE).
