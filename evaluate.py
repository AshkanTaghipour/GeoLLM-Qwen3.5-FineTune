"""
evaluate.py — Post-training evaluation for the fine-tuned Qwen 3.5-0.8B model.

Runs inference on all test splits (QA, CoT, hallucination traps) and computes
standard, widely-accepted NLP metrics, then logs everything to WandB as
summary scores + interactive tables.

Metrics computed (all standard in NLP evaluation literature):
  - ROUGE-1/2/L (via Google's rouge_score package): The standard metric for
    text generation evaluation. ROUGE-1 = unigram overlap, ROUGE-2 = bigram
    overlap, ROUGE-L = longest common subsequence. All report F1.
    Used in: SQuAD, CNN/DailyMail, most LLM evaluation papers.

  - BERTScore (via bert_score package): Semantic similarity using contextual
    embeddings (DeBERTa). More robust than word-overlap metrics because it
    captures paraphrases and synonyms. Standard in modern NLG evaluation.
    Used in: WMT, GEM benchmark, most recent LLM papers.

  - CoT structure rate: Whether the model produces <think>...</think> tags
    for chain-of-thought examples (domain-specific to Qwen 3.5).

  - Hallucination trap pass rate: Whether the model avoids fabricating info
    or corrects wrong geological premises (domain-specific safety metric).

Usage:
    python evaluate.py                          # uses defaults
    python evaluate.py --model_dir ./finetuned_lora
    python evaluate.py --run_name "eval-epoch3" # custom WandB run name
"""

import argparse
import json
import os
import re
import time

import torch
import wandb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen 3.5-0.8B.")
    p.add_argument("--model_dir", default="./finetuned_lora",
                    help="Path to LoRA adapter directory.")
    p.add_argument("--base_model", default="./models/Qwen3.5-0.8B",
                    help="Path to base model (needed for LoRA loading).")
    p.add_argument("--data_dir", default="./training_splits_v2",
                    help="Directory containing test JSONL files.")
    p.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max tokens to generate per response.")
    p.add_argument("--run_name", default=None,
                    help="WandB run name (auto-generated if omitted).")
    p.add_argument("--wandb_project", default="qwen35-geology-finetune",
                    help="WandB project name.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


SYSTEM_PROMPT = (
    "You are a specialist geologist and exploration consultant with over "
    "10 years of experience in Western Australian and Queensland mineral "
    "exploration. You provide expert advice on geological interpretation, "
    "exploration methods, deposit models, geochemistry, geophysics, and "
    "drilling strategies. You answer like a knowledgeable colleague — concise, "
    "technically specific, and grounded in real geological data."
)


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------

def compute_rouge_scores(predictions, references):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L using Google's rouge_score package.
    Returns per-example dicts and averaged scores.
    This is the same implementation used in the HuggingFace evaluate library.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    per_example = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        per_example.append({
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        })

    # Average across examples
    avg = {}
    for key in ["rouge1", "rouge2", "rougeL"]:
        avg[key] = sum(d[key] for d in per_example) / max(len(per_example), 1)

    return per_example, avg


def compute_bert_scores(predictions, references):
    """
    Compute BERTScore using the bert_score package (DeBERTa-based).
    BERTScore computes token-level cosine similarity between contextual
    embeddings, then aggregates into precision, recall, F1.
    This is the standard semantic similarity metric in NLG evaluation.
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    per_example = [
        {"precision": p.item(), "recall": r.item(), "f1": f.item()}
        for p, r, f in zip(P, R, F1)
    ]
    avg_f1 = F1.mean().item()

    return per_example, avg_f1


def check_think_tags(text):
    """Check if response contains proper <think>...</think> structure."""
    return "<think>" in text and "</think>" in text


def check_hallucination_trap(response, acceptable_responses):
    """
    Check if the model's response aligns with any acceptable behavior.
    Uses keyword matching against the acceptable_responses descriptions.
    Returns (passed: bool, reason: str).
    """
    response_lower = response.lower()

    caution_keywords = [
        "don't recognise", "don't recognize", "not familiar",
        "cannot confirm", "can't confirm", "not aware",
        "no record", "couldn't find", "unable to verify",
        "clarif", "could you clarif", "do you mean",
        "i'm not sure", "not a recognised", "not a recognized",
        "doesn't exist", "does not exist", "fictional",
        "no deposit", "unknown deposit",
    ]

    correction_keywords = [
        "however", "actually", "correction", "not quite",
        "doesn't occur", "don't form", "does not occur",
        "wouldn't expect", "misconception", "incorrect",
        "not typical", "not the right", "not appropriate",
        "rather than", "instead",
    ]

    for keyword in caution_keywords:
        if keyword in response_lower:
            return True, f"Model showed caution (matched: '{keyword}')"

    for keyword in correction_keywords:
        if keyword in response_lower:
            return True, f"Model corrected premise (matched: '{keyword}')"

    return False, "Model may have hallucinated — no caution or correction detected"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, question, max_new_tokens=1024,
                       enable_thinking=False):
    """Generate a response for a single question using the fine-tuned model.

    Note: Qwen 3.5-0.8B is a VL (vision-language) model, so `tokenizer` is
    actually a Qwen3VLProcessor. Calling tokenizer(text) directly triggers the
    image processor and crashes. Instead, we use apply_chat_template with
    tokenize=False, then tokenize via the underlying text tokenizer.

    Args:
        enable_thinking: If True, the Qwen 3.5 chat template opens a <think>
            block and lets the model generate reasoning. If False, the template
            inserts an empty <think></think> block which tells the model to
            skip reasoning and answer directly.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Get the chat-formatted text first, then tokenize via the underlying
    # text tokenizer (not the VL processor which crashes on text-only input).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Access the underlying text tokenizer to avoid the VL image processor
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = text_tok(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )

    generated_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    model_dir="./finetuned_lora",
    base_model="./models/Qwen3.5-0.8B",
    data_dir="./training_splits",
    max_new_tokens=1024,
    run_name=None,
    wandb_project="qwen35-geology-finetune",
    model=None,
    tokenizer=None,
    log_to_wandb=True,
):
    """
    Run full evaluation. Can be called from train.py (pass model/tokenizer
    directly to avoid reloading) or standalone.
    """
    os.environ["HF_HOME"] = os.path.abspath("./models")
    os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("./models")

    # ------------------------------------------------------------------
    # 1. Load model (skip if already provided by train.py)
    # ------------------------------------------------------------------
    from unsloth import FastLanguageModel

    if model is None or tokenizer is None:
        print(f"[eval] Loading LoRA adapters from {model_dir} ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_16bit=True,
        )

    FastLanguageModel.for_inference(model)
    print("[eval] Model set to inference mode.")

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    qa_test = load_jsonl(os.path.join(data_dir, "qa_test.jsonl"))
    cot_test = load_jsonl(os.path.join(data_dir, "cot_test.jsonl"))
    traps = load_jsonl(os.path.join(data_dir, "hallucination_traps.jsonl"))
    print(f"[eval] Test data: {len(qa_test)} QA, {len(cot_test)} CoT, "
          f"{len(traps)} hallucination traps")

    # ------------------------------------------------------------------
    # 3. Initialize WandB (skip if log_to_wandb=False)
    # ------------------------------------------------------------------
    if log_to_wandb:
        run_name = run_name or f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            job_type="evaluation",
            config={
                "model_dir": model_dir,
                "base_model": base_model,
                "max_new_tokens": max_new_tokens,
                "qa_test_size": len(qa_test),
                "cot_test_size": len(cot_test),
                "trap_count": len(traps),
            },
        )

    # ------------------------------------------------------------------
    # 4. Generate responses for QA test set (thinking OFF — direct answers)
    # ------------------------------------------------------------------
    print("\n[eval] === Generating QA responses (thinking=OFF) ===")
    qa_predictions = []
    qa_references = []
    for i, ex in enumerate(qa_test):
        print(f"  QA {i+1}/{len(qa_test)}: {ex['question'][:60]}...")
        resp = generate_response(model, tokenizer, ex["question"], max_new_tokens,
                                 enable_thinking=False)
        qa_predictions.append(resp)
        qa_references.append(ex["answer"])

    # ------------------------------------------------------------------
    # 5. Generate responses for CoT test set (thinking ON — model reasons)
    # ------------------------------------------------------------------
    print("\n[eval] === Generating CoT responses (thinking=ON) ===")
    cot_predictions_full = []  # full response including <think> tags
    cot_predictions_answer = []  # answer portion only (after </think>)
    cot_references = []
    cot_reasoning_refs = []
    cot_think_tags = []

    for i, ex in enumerate(cot_test):
        print(f"  CoT {i+1}/{len(cot_test)}: {ex['question'][:60]}...")
        resp = generate_response(model, tokenizer, ex["question"], max_new_tokens,
                                 enable_thinking=True)
        cot_predictions_full.append(resp)

        has_think = check_think_tags(resp)
        cot_think_tags.append(has_think)

        # Extract answer portion (after </think>) for metric comparison
        if "</think>" in resp:
            answer_part = resp.split("</think>", 1)[1].strip()
        else:
            answer_part = resp
        cot_predictions_answer.append(answer_part)
        cot_references.append(ex["answer"])
        cot_reasoning_refs.append(ex.get("reasoning", ""))

    # ------------------------------------------------------------------
    # 6. Generate responses for hallucination traps
    # ------------------------------------------------------------------
    print("\n[eval] === Generating hallucination trap responses ===")
    trap_responses = []
    trap_results = []

    for i, trap in enumerate(traps):
        print(f"  Trap {i+1}/{len(traps)} ({trap['category']}): "
              f"{trap['question'][:50]}...")
        resp = generate_response(model, tokenizer, trap["question"], max_new_tokens)
        trap_responses.append(resp)

        passed, reason = check_hallucination_trap(resp, trap["acceptable_responses"])
        trap_results.append({"passed": passed, "reason": reason})
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {reason}")

    # ------------------------------------------------------------------
    # 7. Compute standard metrics
    # ------------------------------------------------------------------
    print("\n[eval] === Computing ROUGE scores ===")
    qa_rouge_per, qa_rouge_avg = compute_rouge_scores(qa_predictions, qa_references)
    cot_rouge_per, cot_rouge_avg = compute_rouge_scores(
        cot_predictions_answer, cot_references
    )

    print("[eval] === Computing BERTScore (this may take a moment) ===")
    qa_bert_per, qa_bert_avg = compute_bert_scores(qa_predictions, qa_references)
    cot_bert_per, cot_bert_avg = compute_bert_scores(
        cot_predictions_answer, cot_references
    )

    # CoT-specific metrics
    think_tag_count = sum(cot_think_tags)
    think_tag_rate = think_tag_count / max(len(cot_test), 1)

    # Hallucination metrics
    trap_pass_count = sum(1 for r in trap_results if r["passed"])
    trap_pass_rate = trap_pass_count / max(len(traps), 1)

    # ------------------------------------------------------------------
    # 8. Build summary
    # ------------------------------------------------------------------
    summary = {
        # QA — ROUGE (standard text overlap)
        "qa/rouge1_f1": qa_rouge_avg["rouge1"],
        "qa/rouge2_f1": qa_rouge_avg["rouge2"],
        "qa/rougeL_f1": qa_rouge_avg["rougeL"],
        # QA — BERTScore (semantic similarity)
        "qa/bertscore_f1": qa_bert_avg,
        "qa/count": len(qa_test),

        # CoT — ROUGE (on answer portion)
        "cot/rouge1_f1": cot_rouge_avg["rouge1"],
        "cot/rouge2_f1": cot_rouge_avg["rouge2"],
        "cot/rougeL_f1": cot_rouge_avg["rougeL"],
        # CoT — BERTScore
        "cot/bertscore_f1": cot_bert_avg,
        # CoT — structure
        "cot/think_tag_rate": think_tag_rate,
        "cot/count": len(cot_test),

        # Hallucination safety
        "hallucination/pass_rate": trap_pass_rate,
        "hallucination/passed": trap_pass_count,
        "hallucination/total": len(traps),

        # Overall weighted score
        # 35% QA ROUGE-L + 25% QA BERTScore + 20% CoT ROUGE-L + 10% CoT think tags + 10% trap pass
        "overall/weighted_score": (
            0.35 * qa_rouge_avg["rougeL"]
            + 0.25 * qa_bert_avg
            + 0.20 * cot_rouge_avg["rougeL"]
            + 0.10 * think_tag_rate
            + 0.10 * trap_pass_rate
        ),
    }

    # ------------------------------------------------------------------
    # 9. Log to WandB (skip if log_to_wandb=False)
    # ------------------------------------------------------------------
    if log_to_wandb:
        # Summary metrics (appear as numbers in the WandB overview)
        for key, val in summary.items():
            wandb.run.summary[key] = round(val, 4) if isinstance(val, float) else val

        # --- Summary metrics table (aggregated scores in one place) ---
        summary_table = wandb.Table(columns=[
            "category", "n_examples", "ROUGE-1", "ROUGE-2", "ROUGE-L",
            "BERTScore_F1", "think_tag_rate", "hallucination_pass", "weighted_score",
        ])
        summary_table.add_data(
            "QA", len(qa_test),
            round(qa_rouge_avg["rouge1"], 4),
            round(qa_rouge_avg["rouge2"], 4),
            round(qa_rouge_avg["rougeL"], 4),
            round(qa_bert_avg, 4),
            None, None, None,
        )
        summary_table.add_data(
            "CoT (answer)", len(cot_test),
            round(cot_rouge_avg["rouge1"], 4),
            round(cot_rouge_avg["rouge2"], 4),
            round(cot_rouge_avg["rougeL"], 4),
            round(cot_bert_avg, 4),
            round(think_tag_rate, 4),
            None, None,
        )
        summary_table.add_data(
            "Hallucination", len(traps),
            None, None, None, None, None,
            round(trap_pass_rate, 4), None,
        )
        summary_table.add_data(
            "OVERALL", len(qa_test) + len(cot_test) + len(traps),
            None, None, None, None, None, None,
            round(summary["overall/weighted_score"], 4),
        )

        # --- Per-example QA results table ---
        qa_table = wandb.Table(columns=[
            "question", "reference", "generated",
            "rouge1", "rouge2", "rougeL", "bertscore_f1",
        ])
        for i, ex in enumerate(qa_test):
            qa_table.add_data(
                ex["question"], ex["answer"], qa_predictions[i],
                round(qa_rouge_per[i]["rouge1"], 4),
                round(qa_rouge_per[i]["rouge2"], 4),
                round(qa_rouge_per[i]["rougeL"], 4),
                round(qa_bert_per[i]["f1"], 4),
            )

        # --- Per-example CoT results table ---
        cot_table = wandb.Table(columns=[
            "question", "ref_answer", "generated", "has_think_tags",
            "rouge1", "rouge2", "rougeL", "bertscore_f1",
        ])
        for i, ex in enumerate(cot_test):
            cot_table.add_data(
                ex["question"], ex["answer"], cot_predictions_full[i],
                cot_think_tags[i],
                round(cot_rouge_per[i]["rouge1"], 4),
                round(cot_rouge_per[i]["rouge2"], 4),
                round(cot_rouge_per[i]["rougeL"], 4),
                round(cot_bert_per[i]["f1"], 4),
            )

        # --- Per-example hallucination trap table ---
        trap_table = wandb.Table(columns=[
            "id", "category", "question", "trap_detail",
            "generated", "passed", "reason",
        ])
        for i, trap in enumerate(traps):
            trap_table.add_data(
                trap["id"], trap["category"], trap["question"],
                trap["trap_detail"], trap_responses[i],
                trap_results[i]["passed"], trap_results[i]["reason"],
            )

        # Log ALL tables in a single wandb.log() call
        wandb.log({
            "eval/summary": summary_table,
            "eval/qa_results": qa_table,
            "eval/cot_results": cot_table,
            "eval/hallucination_traps": trap_table,
        })

    # ------------------------------------------------------------------
    # 10. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  QA Test ({len(qa_test)} examples):")
    print(f"    ROUGE-1 F1:    {summary['qa/rouge1_f1']:.4f}")
    print(f"    ROUGE-2 F1:    {summary['qa/rouge2_f1']:.4f}")
    print(f"    ROUGE-L F1:    {summary['qa/rougeL_f1']:.4f}")
    print(f"    BERTScore F1:  {summary['qa/bertscore_f1']:.4f}")

    print(f"\n  CoT Test ({len(cot_test)} examples):")
    print(f"    ROUGE-1 F1:    {summary['cot/rouge1_f1']:.4f}")
    print(f"    ROUGE-2 F1:    {summary['cot/rouge2_f1']:.4f}")
    print(f"    ROUGE-L F1:    {summary['cot/rougeL_f1']:.4f}")
    print(f"    BERTScore F1:  {summary['cot/bertscore_f1']:.4f}")
    print(f"    Think tags:    {think_tag_count}/{len(cot_test)} "
          f"({think_tag_rate:.0%})")

    print(f"\n  Hallucination Traps ({len(traps)} traps):")
    print(f"    Pass rate:     {trap_pass_count}/{len(traps)} "
          f"({trap_pass_rate:.0%})")

    print(f"\n  Overall Weighted Score: {summary['overall/weighted_score']:.4f}")
    print(f"    (35% QA ROUGE-L + 25% QA BERTScore + 20% CoT ROUGE-L"
          f" + 10% think-tag + 10% trap-pass)")

    if log_to_wandb and wandb.run is not None:
        print(f"\n  WandB dashboard: {wandb.run.get_url()}")
        wandb.finish()

    print("=" * 60)

    return summary


def main():
    """CLI entry point — parses args and calls run_evaluation()."""
    args = parse_args()
    return run_evaluation(
        model_dir=args.model_dir,
        base_model=args.base_model,
        data_dir=args.data_dir,
        max_new_tokens=args.max_new_tokens,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
