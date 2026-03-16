# GeoLLM-Qwen3.5-FineTune

**Domain-adapted LLMs for mineral exploration geology** -- fine-tuning and benchmarking the full Qwen 3.5 model family (0.8B to 27B) on Western Australian geological knowledge.

## Why This Project?

General-purpose LLMs know surprisingly little about the specifics of mineral exploration -- the deposit models, pathfinder geochemistry, geophysical targeting methods, and drilling strategies that working geologists rely on daily. This project explores whether **parameter-efficient fine-tuning** (LoRA) can inject real domain expertise from exploration reports into open-weight models, and how that capability scales across five model sizes.

The goal: an LLM that answers like a knowledgeable colleague -- concise, technically grounded, and fluent in Western Australian geology -- rather than a generic chatbot that hallucinates geological facts.

---

## Dataset

The training data was constructed from **~300 recent mineral exploration reports** sourced from [WAMEX](https://www.dmp.wa.gov.au/WAMEX-Minerals-Exploration-1476.aspx) (Western Australia Mineral Exploration Index), a public repository maintained by the WA Department of Mines.

**How it was built:**
1. PDF reports were OCR-processed to extract full text alongside structured abstracts
2. An LLM read each report and generated expert-style QA and chain-of-thought pairs -- the kind of questions a mining geologist or exploration manager would ask a colleague
3. All report identifiers and company names were stripped; answers read as domain expertise, not document summaries
4. Each batch was human-reviewed for geological accuracy and de-duplicated against existing pairs
5. 15 hallucination traps were hand-curated with fabricated deposits, impossible assay values, and wrong geological provinces

| Split | QA Pairs | CoT Pairs | Total |
|:------|:---------|:----------|:------|
| Train | 352 | 127 | 479 |
| Test | 19 | 7 | 26 |
| Hallucination traps | -- | -- | 15 |

**Topics covered:** orogenic gold, komatiite nickel, VMS base metals, lithium pegmatites, iron ore, IOCG copper-gold, geophysics (magnetics, gravity, EM, IP), geochemistry (pathfinders, soil sampling), drilling methods, structural geology, deposit models, and regional WA geology (Yilgarn, Pilbara, Murchison, Gascoyne, Albany-Fraser).

📦 **Dataset on HuggingFace:** [AshkanTaghipour/mineral-exploration-geology-qa](https://huggingface.co/datasets/AshkanTaghipour/mineral-exploration-geology-qa)

---

## Benchmark Results

All five models were trained for **5 epochs** with **bf16 LoRA** (r=16, alpha=16) on a single NVIDIA A100-80GB GPU using the same dataset and hyperparameters. Each model was evaluated **before and after** fine-tuning on the same held-out test set.

### QA Metrics (19 test examples)

| Model | Params | QA ROUGE-L (base) | QA ROUGE-L (FT) | QA BERTScore (base) | QA BERTScore (FT) |
|:------|-------:|-------------------:|-----------------:|--------------------:|------------------:|
| Qwen3.5-0.8B | 0.8B | 0.142 | 0.170 | 0.812 | 0.845 |
| Qwen3.5-2B | 2.0B | 0.134 | 0.187 | 0.811 | 0.848 |
| Qwen3.5-4B | 4.0B | 0.130 | 0.193 | 0.814 | 0.853 |
| Qwen3.5-9B | 9.0B | 0.137 | 0.195 | 0.819 | 0.853 |
| Qwen3.5-27B | 27.0B | 0.145 | **0.194** | 0.819 | **0.853** |

### Chain-of-Thought Metrics (7 test examples)

| Model | CoT ROUGE-L (base) | CoT ROUGE-L (FT) | CoT BERTScore (base) | CoT BERTScore (FT) |
|:------|-------------------:|-----------------:|---------------------:|-------------------:|
| Qwen3.5-0.8B | 0.128 | 0.203 | 0.783 | 0.855 |
| Qwen3.5-2B | 0.126 | 0.197 | 0.789 | 0.851 |
| Qwen3.5-4B | 0.127 | 0.195 | 0.789 | 0.858 |
| Qwen3.5-9B | 0.125 | 0.206 | 0.795 | 0.862 |
| Qwen3.5-27B | 0.137 | **0.234** | 0.799 | **0.861** |

### Hallucination Safety (15 traps)

| Model | Pass Rate (base) | Pass Rate (FT) | Delta |
|:------|-----------------:|---------------:|------:|
| Qwen3.5-0.8B | 66.7% | 40.0% | -26.7% |
| Qwen3.5-2B | 80.0% | 26.7% | -53.3% |
| Qwen3.5-4B | 66.7% | 33.3% | -33.3% |
| Qwen3.5-9B | 73.3% | 20.0% | -53.3% |
| Qwen3.5-27B | 60.0% | 33.3% | -26.7% |

### Training Summary

| Model | Train Loss | Wall Time | Trainable Params | Trainable % |
|:------|----------:|---------:|-----------------:|------------:|
| Qwen3.5-0.8B | 1.863 | 15 min | 6.4M | 0.74% |
| Qwen3.5-2B | 1.576 | 15 min | 10.9M | 0.49% |
| Qwen3.5-4B | 1.316 | 22 min | 21.2M | 0.47% |
| Qwen3.5-9B | 1.172 | 37 min | 29.1M | 0.31% |
| Qwen3.5-27B | 1.005 | 133 min | 79.7M | 0.29% |

📊 **Live training curves and comparison charts:** [WandB Dashboard](https://wandb.ai/ash-developer-2023-the-university-of-western-australia/qwen35-geology-finetune?nw=nwuserashdeveloper2023)

### Key Findings

- 🏆 **27B achieves the best fine-tuned performance**, with the highest CoT ROUGE-L (0.234) and the best overall weighted score (0.361), benefiting most from domain adaptation
- 📉 **Training loss scales inversely with model size** (1.86 for 0.8B down to 1.00 for 27B), showing larger models fit geological language more efficiently
- 📈 **ROUGE and BERTScore improve across all models** after fine-tuning -- QA BERTScore gains ~0.03-0.04 across the board, confirming that LoRA successfully adapts responses toward expert geological language
- ⚠️ **Hallucination resistance degrades after fine-tuning** for all models (average -39% pass rate). Fine-tuned models become more fluent and assertive in the geology domain, but less likely to flag fabricated premises -- a critical consideration for deployment
- ⚡ **0.8B offers the best cost-efficiency**: competitive QA scores in just 15 minutes of training, suitable for deployment on consumer GPUs with ~2 GB VRAM

### How Are These Metrics Calculated?

- **ROUGE-L** measures the longest common subsequence between generated and reference answers -- higher means more word-level overlap with expert responses
- **BERTScore** measures semantic similarity using contextual embeddings (DeBERTa) -- captures paraphrases and synonyms that ROUGE misses
- **Hallucination pass rate** tests whether the model flags fabricated geological entities (fictional deposits, impossible assay values, wrong provinces) rather than confidently answering them

For the full methodology, metric formulas, and hallucination trap categories, see the [Evaluation Methodology](docs/EVALUATION.md). For complete per-metric breakdowns across all models, see the [Benchmark Analysis](docs/BENCHMARK.md).

---

## Models

All fine-tuned models are merged bf16 checkpoints on HuggingFace -- ready for direct inference with `transformers`, no adapter loading required:

| Model | HuggingFace | VRAM (inference) |
|:------|:------------|:-----------------|
| GeoLLM-Qwen3.5-0.8B | [AshkanTaghipour/GeoLLM-Qwen3.5-0.8B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-0.8B) | ~2 GB |
| GeoLLM-Qwen3.5-2B | [AshkanTaghipour/GeoLLM-Qwen3.5-2B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-2B) | ~5 GB |
| GeoLLM-Qwen3.5-4B | [AshkanTaghipour/GeoLLM-Qwen3.5-4B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-4B) | ~9 GB |
| GeoLLM-Qwen3.5-9B | [AshkanTaghipour/GeoLLM-Qwen3.5-9B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-9B) | ~19 GB |
| GeoLLM-Qwen3.5-27B | [AshkanTaghipour/GeoLLM-Qwen3.5-27B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-27B) | ~55 GB |

---

## Quick Start

### Inference

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

Or use the standalone inference script:

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-4B --question "What are the key pathfinder elements for orogenic gold in the Yilgarn Craton?"
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-4B --interactive
```

### Reproduce Training

```bash
# 1. Set up environment
bash setup_env.sh

# 2. Prepare data
python prepare_data.py --input_dir ./training_splits_v2 --output_dir ./processed_data

# 3. Train a single model
python train.py --model_name unsloth/Qwen3.5-0.8B --epochs 5

# 4. Run the full benchmark (all 5 models sequentially)
python benchmark.py --epochs 5

# 5. Evaluate a fine-tuned model
python evaluate.py --model_dir ./finetuned_lora --run_name "my-eval"
```

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
    ├── BENCHMARK.md          # Full benchmark tables and analysis
    └── EVALUATION.md         # Evaluation methodology and metric definitions
```

---

## Documentation

- **[Benchmark Results and Analysis](docs/BENCHMARK.md)** -- Complete per-metric tables, training statistics, and five key findings across all model sizes
- **[Evaluation Methodology](docs/EVALUATION.md)** -- Metric definitions, hallucination trap categories, weighted scoring formula, and inference settings

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
