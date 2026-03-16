# Data Preparation Guide

How raw geology QA/CoT data is transformed into training-ready format for Qwen 3.5-0.8B fine-tuning.

## Overview

The pipeline (`prepare_data.py`) takes raw JSONL files and produces:
1. **Split JSONL files** — `qa_train.jsonl`, `qa_test.jsonl`, `cot_train.jsonl`, `cot_test.jsonl` (used by `evaluate.py`)
2. **HuggingFace DatasetDict** — Arrow-format dataset in `processed_data/` (used by `train.py`)
3. **Hallucination traps** — `hallucination_traps.jsonl` (hand-curated, used by `evaluate.py`)

## Input Formats

The script auto-detects two input layouts:

### Combined format (v2 — current)
```
training_splits_v2/
├── qa_pairs.jsonl              # All QA pairs in one file
├── cot_pairs.jsonl             # All CoT pairs in one file
└── hallucination_traps.jsonl   # Hand-curated evaluation traps
```

- **QA records**: `{"question": "...", "answer": "..."}`
- **CoT records**: `{"question": "...", "reasoning": "...", "answer": "..."}`
- Extra metadata keys (e.g., `batch_id`, `source_anumbers`) are ignored during processing

### Pre-split format (v1 — legacy)
```
training_splits/
├── qa_train.jsonl / qa_test.jsonl
├── cot_train.jsonl / cot_test.jsonl
└── hallucination_traps.jsonl
```

When pre-split files are detected, the script skips splitting and uses them directly.

## Processing Steps

### Step 1: Validation

Every record is checked for required fields:

| Data type | Required fields |
|-----------|----------------|
| QA | `question`, `answer` (both non-empty) |
| CoT | `question`, `reasoning`, `answer` (all non-empty) |

**Handling invalid records:**
- CoT records missing only `reasoning` are **demoted to QA** (the Q&A content is still valuable)
- Records missing `question` or `answer` are **dropped** with a warning

### Step 2: Stratified Train/Test Split

When using combined-format input, the data is split using **stratified sampling**:

1. QA and CoT records are split **independently** at the same ratio
2. This guarantees both data types appear proportionally in train and test
3. Critical when test sets are small — prevents accidental skew

**Default configuration:**
- **Split ratio**: 95% train / 5% test
- **Random seed**: 42 (for reproducibility)

**Why stratified?** With a 95/5 split and ~500 examples, the test set has only ~25 records. Without stratification, random chance could leave the test set with very few CoT examples (or none), making CoT evaluation unreliable.

### Step 3: Chat Format Conversion

Each record is converted to the Qwen 3.5 chat message format:

**QA example → chat format:**
```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<answer>"}
  ],
  "data_type": "qa"
}
```

**CoT example → chat format:**
```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<think>\n<reasoning>\n</think>\n\n<answer>"}
  ],
  "data_type": "cot"
}
```

The `<think>` tags match Qwen 3.5's native thinking mode. Training with this format teaches the model when and how to reason before answering.

### Step 4: Shuffling

- **Train set**: shuffled (prevents catastrophic forgetting from seeing all QA then all CoT)
- **Test set**: NOT shuffled (deterministic order for reproducible evaluation)

### Step 5: Save

Two outputs are written:

1. **Split JSONL files** → saved to the input directory (e.g., `training_splits_v2/`)
   - Clean keys only (no metadata like `batch_id`)
   - Used by `evaluate.py` for inference and metric computation
2. **HuggingFace DatasetDict** → saved to `processed_data/`
   - Apache Arrow format for efficient batched loading
   - Used by `train.py` via SFTTrainer

## Running the Pipeline

### Standard run (v2 data, 95/5 split)
```bash
python prepare_data.py
```

### Custom split ratio
```bash
python prepare_data.py --test_ratio 0.10 --seed 123
```

### Using v1 data (pre-split, no splitting needed)
```bash
python prepare_data.py --input_dir ./training_splits
```

### Full argument list
```bash
python prepare_data.py \
    --input_dir ./training_splits_v2 \
    --output_dir ./processed_data \
    --test_ratio 0.05 \
    --seed 42
```

## After Preparation: Training and Evaluation

```bash
# Train (reads from processed_data/)
python train.py

# Evaluate (reads split JSONL + hallucination traps from training_splits_v2/)
python evaluate.py --data_dir ./training_splits_v2
```

## Adding New Data

When you receive new data:

1. **Replace** `training_splits_v2/qa_pairs.jsonl` and/or `cot_pairs.jsonl` with the updated files
2. **Re-run** `python prepare_data.py` — it will re-validate, re-split, and regenerate everything
3. **Review** the printed statistics to verify counts and catch any validation warnings
4. The split JSONL files (`qa_train.jsonl`, etc.) will be overwritten with the new split

**Important:** Each run with the same `--seed` produces the same split. Change the seed if you want a different random split for cross-validation or robustness checks.

## Hallucination Traps

Hallucination traps are a separate, hand-curated evaluation set. They test whether the model avoids fabricating information or corrects wrong premises.

**Current trap categories (15 traps):**

| # | Category | Tests for |
|---|----------|-----------|
| 1 | `fictional_deposit` | Inventing knowledge about non-existent deposits |
| 2 | `wrong_geology` | Accepting geological misconceptions (VMS in granite) |
| 3 | `impossible_values` | Accepting unrealistic assay values |
| 4 | `fictional_method` | Pretending to know fictional geophysical methods |
| 5 | `wrong_province` | Misattributing deposit types to wrong geological provinces |
| 6 | `invented_formation` | Accepting fictional geological formation names |
| 7 | `report_fishing` | Pretending to have access to specific WAMEX reports |
| 8 | `temporal_impossibility` | Fabricating details about non-existent future technology |
| 9 | `wrong_commodity` | Not flagging unusual commodity-province combinations |
| 10 | `fabricated_statistic` | Accepting made-up statistics without pushback |
| 11 | `wrong_regulation` | Accepting incorrect regulatory/legal claims about mining |
| 12 | `wrong_pathfinder` | Accepting wrong geochemical pathfinder associations |
| 13 | `wrong_deposit_model` | Placing deposit types in geologically impossible basins |
| 14 | `fabricated_resource` | Confirming specific JORC resource figures it cannot verify |
| 15 | `wrong_method_for_commodity` | Accepting inappropriate geophysical methods for a target type |

Hallucination traps are **not** part of the train/test split — they are always used for evaluation only. To add new traps, append records to `training_splits_v2/hallucination_traps.jsonl`.

## Current Dataset Statistics (v2, seed=42, 95/5 split)

| | QA | CoT | Total |
|---|---|---|---|
| **Train** | 352 | 127 | 479 |
| **Test** | 19 | 7 | 26 |
| **Hallucination traps** | — | — | 15 |
| **Total** | 371 | 134 | 505 (+15 traps) |

- 1 CoT record was demoted to QA (missing `reasoning` field)
- Average message length: ~2,130 chars (within the 2048-token `max_seq_length`)

## Reproducibility

The split is fully deterministic given the same:
- Input files (same records in the same order)
- `--test_ratio` (default 0.05)
- `--seed` (default 42)

Re-running with identical inputs and parameters will always produce the same train/test split.
