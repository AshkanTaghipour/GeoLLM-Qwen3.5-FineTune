# Benchmark Results and Analysis

Comprehensive benchmark of five Qwen 3.5 dense models (0.8B -- 27B parameters), each fine-tuned with identical LoRA hyperparameters on the same mineral exploration geology dataset.

**Training configuration:** 5 epochs, bf16 LoRA (r=16, alpha=16), adamw_8bit optimizer, cosine LR schedule (2e-4, 10% warmup), effective batch size 16, max sequence length 2048. All experiments run on a single NVIDIA A100-80GB GPU.

---

## Overall Scores

| Model | Params | Base | Fine-tuned | Delta |
|:------|-------:|-----:|-----------:|------:|
| Qwen3.5-0.8B | 0.8B | 0.345 | 0.351 | +0.006 |
| Qwen3.5-2B | 2.0B | 0.355 | 0.343 | -0.012 |
| Qwen3.5-4B | 4.0B | 0.341 | 0.353 | +0.012 |
| Qwen3.5-9B | 9.0B | 0.351 | 0.343 | -0.008 |
| Qwen3.5-27B | 27.0B | 0.343 | **0.361** | **+0.018** |

> Overall weighted score = 35% QA ROUGE-L + 25% QA BERTScore + 20% CoT ROUGE-L + 10% think-tag rate + 10% hallucination pass rate.

---

## QA Evaluation (19 test examples, thinking OFF)

### ROUGE Scores (F1)

| Model | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:------|:-----|--------:|--------:|--------:|
| Qwen3.5-0.8B | base | 0.2809 | 0.0515 | 0.1420 |
| Qwen3.5-0.8B | finetuned | 0.3568 | 0.0643 | 0.1697 |
| Qwen3.5-2B | base | 0.2765 | 0.0467 | 0.1336 |
| Qwen3.5-2B | finetuned | 0.3891 | 0.0819 | 0.1869 |
| Qwen3.5-4B | base | 0.2848 | 0.0485 | 0.1297 |
| Qwen3.5-4B | finetuned | 0.4039 | 0.0865 | 0.1931 |
| Qwen3.5-9B | base | 0.2974 | 0.0543 | 0.1366 |
| Qwen3.5-9B | finetuned | 0.3977 | 0.0914 | 0.1947 |
| Qwen3.5-27B | base | 0.2977 | 0.0574 | 0.1448 |
| Qwen3.5-27B | finetuned | **0.4158** | **0.0901** | 0.1939 |

### BERTScore (F1)

| Model | Base | Fine-tuned | Delta |
|:------|-----:|-----------:|------:|
| Qwen3.5-0.8B | 0.8120 | 0.8447 | +0.033 |
| Qwen3.5-2B | 0.8108 | 0.8475 | +0.037 |
| Qwen3.5-4B | 0.8143 | 0.8533 | +0.039 |
| Qwen3.5-9B | 0.8192 | 0.8525 | +0.033 |
| Qwen3.5-27B | 0.8191 | **0.8526** | +0.034 |

---

## Chain-of-Thought Evaluation (7 test examples, thinking ON)

### ROUGE Scores (F1)

| Model | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:------|:-----|--------:|--------:|--------:|
| Qwen3.5-0.8B | base | 0.2306 | 0.0423 | 0.1283 |
| Qwen3.5-0.8B | finetuned | 0.3516 | 0.0887 | 0.2028 |
| Qwen3.5-2B | base | 0.2514 | 0.0440 | 0.1261 |
| Qwen3.5-2B | finetuned | 0.3526 | 0.0887 | 0.1972 |
| Qwen3.5-4B | base | 0.2644 | 0.0536 | 0.1270 |
| Qwen3.5-4B | finetuned | 0.3357 | 0.0940 | 0.1953 |
| Qwen3.5-9B | base | 0.2722 | 0.0585 | 0.1250 |
| Qwen3.5-9B | finetuned | 0.3913 | 0.1097 | 0.2064 |
| Qwen3.5-27B | base | 0.2888 | 0.0588 | 0.1373 |
| Qwen3.5-27B | finetuned | **0.4056** | **0.1255** | **0.2335** |

### BERTScore and Think-Tag Rate

| Model | Base BERTScore | FT BERTScore | Think-Tag Rate (base) | Think-Tag Rate (FT) |
|:------|---------------:|-------------:|----------------------:|--------------------:|
| Qwen3.5-0.8B | 0.7832 | 0.8554 | 0% | 0% |
| Qwen3.5-2B | 0.7894 | 0.8510 | 0% | 0% |
| Qwen3.5-4B | 0.7888 | 0.8579 | 0% | 0% |
| Qwen3.5-9B | 0.7953 | 0.8619 | 0% | 0% |
| Qwen3.5-27B | 0.7992 | **0.8606** | 0% | 0% |

---

## Hallucination Traps (15 traps)

The hallucination test set contains 15 hand-curated questions with fabricated deposits, impossible values, fictional methods, wrong geological provinces, and other traps designed to test whether the model avoids fabricating information.

| Model | Base Pass Rate | FT Pass Rate | Delta |
|:------|---------------:|-------------:|------:|
| Qwen3.5-0.8B | 66.7% | 40.0% | -26.7% |
| Qwen3.5-2B | 80.0% | 26.7% | -53.3% |
| Qwen3.5-4B | 66.7% | 33.3% | -33.3% |
| Qwen3.5-9B | 73.3% | 20.0% | -53.3% |
| Qwen3.5-27B | 60.0% | 33.3% | -26.7% |

---

## Training Statistics

| Model | Params | Trainable | Trainable % | Train Loss | Wall Time | Batch Size | Grad Accum |
|:------|-------:|----------:|------------:|-----------:|----------:|-----------:|-----------:|
| Qwen3.5-0.8B | 859M | 6.4M | 0.74% | 1.863 | 15 min | 4 | 4 |
| Qwen3.5-2B | 2.2B | 10.9M | 0.49% | 1.576 | 15 min | 4 | 4 |
| Qwen3.5-4B | 4.6B | 21.2M | 0.47% | 1.316 | 22 min | 4 | 4 |
| Qwen3.5-9B | 9.4B | 29.1M | 0.31% | 1.172 | 37 min | 2 | 8 |
| Qwen3.5-27B | 27.4B | 79.7M | 0.29% | 1.005 | 133 min | 1 | 16 |

All models used an effective batch size of 16, achieved by adjusting gradient accumulation steps. The 27B model required the full 80GB of A100 VRAM.

---

## Key Findings

### 1. Larger models achieve lower training loss but the relationship to eval scores is non-linear

Training loss decreases monotonically with model size (1.86 for 0.8B to 1.00 for 27B), indicating larger models fit the geology training data more efficiently. However, the overall evaluation score does not follow this trend linearly -- the 27B model achieves the best fine-tuned score, but the 2B and 9B models show slight regressions compared to their base performance.

### 2. Fine-tuning consistently improves text overlap and semantic similarity

Across all five models, QA ROUGE-L improves by 0.02 -- 0.05 points and QA BERTScore improves by 0.03 -- 0.04 points after fine-tuning. CoT metrics show even larger gains, with ROUGE-L improving by 0.06 -- 0.10 points. This confirms that LoRA fine-tuning successfully adapts the models to produce responses more aligned with expert geological language.

### 3. Hallucination resistance degrades after fine-tuning

This is the most notable finding. All five models show a significant drop in hallucination trap pass rate after fine-tuning (average decrease of 38.7 percentage points). The fine-tuned models become more assertive and domain-fluent, but less likely to flag fabricated geological entities or impossible values. This is a known challenge in domain fine-tuning and an important consideration for deployment in safety-critical geological advisory roles.

### 4. No model produces chain-of-thought reasoning tags

Despite training on CoT examples with `<think>` tags and evaluating with `enable_thinking=True`, none of the models produce `<think>` tags in their responses. This suggests that 5 epochs of LoRA fine-tuning with 127 CoT examples is insufficient to reliably induce structured reasoning behavior in the Qwen 3.5 architecture.

### 5. The 0.8B model offers the best efficiency-to-performance ratio

The 0.8B model trains in 15 minutes and achieves scores within 3% of the best model (27B, which requires 133 minutes). For resource-constrained deployments or rapid iteration, the 0.8B model provides the best trade-off between training cost and geological capability.

---

## Experimental Notes

- The 27B model benchmark was completed across multiple runs due to CUDA OOM errors during the initial training attempt. Base evaluation and fine-tuned evaluation were obtained from separate runs with identical hyperparameters.
- All models were loaded in bf16 (not QLoRA 4-bit), as recommended by Unsloth documentation for Qwen 3.5.
- Models 0.8B through 9B are architecturally vision-language models but were used in text-only mode via `FastLanguageModel`.
