"""
benchmark.py — Multi-model benchmark for the Qwen 3.5 family.

Sequentially trains and evaluates each model in the registry on the same
geology dataset, producing comparable metrics for base and fine-tuned models.

Results are:
  1. Logged to WandB as a comparison table (one run with all models).
  2. Saved locally as JSON for paper-ready plotting.

Usage:
    # Run all models that fit on A100-80GB
    python benchmark.py

    # Run specific models
    python benchmark.py --models Qwen3.5-0.8B Qwen3.5-2B

    # Custom training config
    python benchmark.py --epochs 5 --learning_rate 1e-4 --lora_r 32
"""

import argparse
import gc
import json
import os
import time

import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Qwen 3.5 model family on geology fine-tuning."
    )
    from model_registry import get_all_model_names
    p.add_argument(
        "--models", nargs="+",
        default=get_all_model_names(),
        help="Model names to benchmark. "
             "Default: all 5 dense models "
             "(Qwen3.5-0.8B, 2B, 4B, 9B, 27B).",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument(
        "--data_dir", default="./training_splits_v2",
        help="Directory with raw JSONL test files for evaluation.",
    )
    p.add_argument(
        "--processed_data_dir", default="./processed_data",
        help="Directory with HuggingFace DatasetDict for training.",
    )
    p.add_argument(
        "--results_dir", default="./benchmark_results",
        help="Directory to save JSON results.",
    )
    p.add_argument(
        "--local_model_dir", default="./models",
        help="Local cache directory for model weights.",
    )
    p.add_argument(
        "--wandb_project", default="qwen35-geology-finetune",
        help="WandB project name.",
    )
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--skip_base_eval", action="store_true",
        help="Skip base model evaluation (only train and eval fine-tuned).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, max_seq_length=2048, local_model_dir="./models"):
    """Load a Qwen 3.5 model via Unsloth's FastLanguageModel.

    Sets HF_HOME and TRANSFORMERS_CACHE to keep downloads in the local
    models directory (not ~/.cache).

    When model_path is a HuggingFace/Unsloth ID (e.g. "unsloth/Qwen3.5-2B"),
    Unsloth downloads it into local_model_dir/hub/. When model_path is a
    local directory (e.g. "./models/Qwen3.5-0.8B"), it loads directly.

    Returns:
        (model, tokenizer) tuple.
    """
    abs_model_dir = os.path.abspath(local_model_dir)
    os.makedirs(abs_model_dir, exist_ok=True)
    os.environ["HF_HOME"] = abs_model_dir
    os.environ["TRANSFORMERS_CACHE"] = abs_model_dir

    from unsloth import FastLanguageModel

    print(f"[model] Loading {model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    print(f"[model] Loaded successfully.")
    return model, tokenizer


def apply_lora(model, lora_r=16, lora_alpha=16, max_seq_length=2048):
    """Apply LoRA adapters to a base model.

    Returns:
        Model with LoRA adapters applied.
    """
    from unsloth import FastLanguageModel
    from model_registry import LORA_TARGET_MODULES

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    return model


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def ensure_processed_data(data_dir, processed_data_dir):
    """Ensure the processed HuggingFace dataset exists. Build if needed."""
    if os.path.isdir(processed_data_dir):
        return
    print(f"[benchmark] Processed data not found at {processed_data_dir}. "
          f"Building from {data_dir} ...")
    from prepare_data import build_dataset
    build_dataset(data_dir, processed_data_dir)


def load_and_format_dataset(processed_data_dir, tokenizer):
    """Load the processed dataset and apply the chat template.

    Returns:
        DatasetDict with 'train' and 'test' splits, 'text' column added.
    """
    from datasets import load_from_disk
    from train import make_formatting_func

    dataset = load_from_disk(processed_data_dir)
    formatting_func = make_formatting_func(tokenizer)
    dataset = dataset.map(
        formatting_func,
        batched=True,
        desc="Applying chat template",
    )
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model, tokenizer, dataset, model_name,
    epochs, learning_rate, lora_r, lora_alpha,
    max_seq_length, batch_size, gradient_accumulation_steps,
    output_dir, wandb_project, wandb_group, seed,
):
    """Train a single model with LoRA and return training metrics.

    Manages its own WandB run (via SFTTrainer's report_to="wandb").

    Returns:
        dict with training metrics (final_loss, wall_time, params, etc).
    """
    import wandb
    from trl import SFTTrainer, SFTConfig
    from train import DetailedMetricsCallback

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[train] Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # Initialize WandB for this training run
    wandb.init(
        project=wandb_project,
        name=f"{model_name}-train-ep{epochs}-r{lora_r}-lr{learning_rate}",
        group=wandb_group,
        job_type="training",
        tags=[model_name, "training"],
        config={
            "model_name": model_name,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": "adamw_8bit",
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.1,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
        },
    )

    checkpoint_dir = os.path.join(output_dir, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=epochs,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=50,
        output_dir=checkpoint_dir,
        optim="adamw_8bit",
        seed=seed,
        bf16=True,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        report_to="wandb",
    )

    metrics_callback = DetailedMetricsCallback()

    # Workaround: Unsloth's fix_untrained_tokens crashes on large models
    # (e.g. 27B) that have meta tensors. Patch it to skip gracefully.
    import unsloth_zoo.tokenizer_utils as _token_utils
    _orig_fix = _token_utils.fix_untrained_tokens

    def _safe_fix_untrained_tokens(*args, **kwargs):
        try:
            return _orig_fix(*args, **kwargs)
        except NotImplementedError:
            print(f"[train] Skipping fix_untrained_tokens "
                  f"(meta tensors in {model_name}, non-critical)")

    _token_utils.fix_untrained_tokens = _safe_fix_untrained_tokens
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[metrics_callback],
        )
    finally:
        _token_utils.fix_untrained_tokens = _orig_fix

    print(f"\n[train] Starting training for {model_name} ...")
    train_start = time.time()
    train_result = trainer.train()
    wall_time = time.time() - train_start

    # Extract training metrics
    final_loss = None
    if hasattr(train_result, "training_loss"):
        final_loss = train_result.training_loss
    elif hasattr(train_result, "metrics"):
        final_loss = train_result.metrics.get("training_loss")

    # Log wall time to WandB before it closes
    if wandb.run is not None:
        wandb.run.summary["wall_time_seconds"] = wall_time
        if final_loss is not None:
            wandb.run.summary["final_training_loss"] = final_loss
        wandb.finish()

    training_metrics = {
        "epochs": epochs,
        "final_train_loss": final_loss,
        "wall_time_seconds": round(wall_time, 1),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }

    return training_metrics


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, data_dir, max_new_tokens=512):
    """Run evaluation on a model (base or fine-tuned) without WandB.

    Returns:
        dict with all evaluation metrics.
    """
    from unsloth import FastLanguageModel
    from evaluate import run_evaluation

    FastLanguageModel.for_inference(model)

    summary = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_new_tokens=max_new_tokens,
        log_to_wandb=False,
    )
    return summary


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def flatten_eval_results(summary):
    """Flatten the eval summary dict into a clean metrics dict.

    Converts keys like 'qa/rouge1_f1' -> 'qa_rouge1' for cleaner JSON.
    """
    return {
        "qa_rouge1": round(summary.get("qa/rouge1_f1", 0), 4),
        "qa_rouge2": round(summary.get("qa/rouge2_f1", 0), 4),
        "qa_rougeL": round(summary.get("qa/rougeL_f1", 0), 4),
        "qa_bertscore_f1": round(summary.get("qa/bertscore_f1", 0), 4),
        "cot_rouge1": round(summary.get("cot/rouge1_f1", 0), 4),
        "cot_rouge2": round(summary.get("cot/rouge2_f1", 0), 4),
        "cot_rougeL": round(summary.get("cot/rougeL_f1", 0), 4),
        "cot_bertscore_f1": round(summary.get("cot/bertscore_f1", 0), 4),
        "cot_think_tag_rate": round(summary.get("cot/think_tag_rate", 0), 4),
        "hallucination_pass_rate": round(
            summary.get("hallucination/pass_rate", 0), 4
        ),
        "overall_weighted_score": round(
            summary.get("overall/weighted_score", 0), 4
        ),
    }


def save_model_result(results_dir, model_name, result):
    """Save a single model's results to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[results] Saved {path}")


def save_benchmark_results(results_dir, benchmark_data):
    """Save the aggregate benchmark results to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "benchmark_summary.json")
    with open(path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[results] Saved {path}")


# ---------------------------------------------------------------------------
# GPU memory cleanup
# ---------------------------------------------------------------------------

def cleanup_gpu():
    """Free GPU memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# WandB comparison table
# ---------------------------------------------------------------------------

def log_comparison_to_wandb(benchmark_data, wandb_project, wandb_group):
    """Create a WandB run with a comparison table for all models."""
    import wandb

    wandb.init(
        project=wandb_project,
        name=f"{wandb_group}-comparison",
        group=wandb_group,
        job_type="comparison",
        tags=["comparison", "benchmark"],
    )

    # --- Comparison table ---
    columns = [
        "model", "params_B", "type",
        "qa_rouge1", "qa_rouge2", "qa_rougeL", "qa_bertscore_f1",
        "cot_rouge1", "cot_rouge2", "cot_rougeL", "cot_bertscore_f1",
        "cot_think_tag_rate", "hallucination_pass_rate",
        "overall_weighted_score",
    ]
    table = wandb.Table(columns=columns)

    for model_name, result in benchmark_data["models"].items():
        params_b = result.get("params_b", 0)

        if "base_eval" in result and result["base_eval"] is not None:
            base = result["base_eval"]
            table.add_data(
                model_name, params_b, "base",
                base.get("qa_rouge1", 0), base.get("qa_rouge2", 0),
                base.get("qa_rougeL", 0), base.get("qa_bertscore_f1", 0),
                base.get("cot_rouge1", 0), base.get("cot_rouge2", 0),
                base.get("cot_rougeL", 0), base.get("cot_bertscore_f1", 0),
                base.get("cot_think_tag_rate", 0),
                base.get("hallucination_pass_rate", 0),
                base.get("overall_weighted_score", 0),
            )

        if "finetuned_eval" in result and result["finetuned_eval"] is not None:
            ft = result["finetuned_eval"]
            table.add_data(
                model_name, params_b, "finetuned",
                ft.get("qa_rouge1", 0), ft.get("qa_rouge2", 0),
                ft.get("qa_rougeL", 0), ft.get("qa_bertscore_f1", 0),
                ft.get("cot_rouge1", 0), ft.get("cot_rouge2", 0),
                ft.get("cot_rougeL", 0), ft.get("cot_bertscore_f1", 0),
                ft.get("cot_think_tag_rate", 0),
                ft.get("hallucination_pass_rate", 0),
                ft.get("overall_weighted_score", 0),
            )

    wandb.log({"benchmark/comparison": table})

    # Also log individual metrics for bar chart visualization
    for model_name, result in benchmark_data["models"].items():
        for eval_type in ["base_eval", "finetuned_eval"]:
            if eval_type in result and result[eval_type] is not None:
                tag = "base" if eval_type == "base_eval" else "finetuned"
                prefix = f"{model_name}/{tag}"
                for metric_key, metric_val in result[eval_type].items():
                    wandb.run.summary[f"{prefix}/{metric_key}"] = metric_val

    url = wandb.run.get_url() if wandb.run else None
    wandb.finish()
    return url


# ---------------------------------------------------------------------------
# Comparison table (terminal)
# ---------------------------------------------------------------------------

def print_comparison_table(benchmark_data):
    """Print a formatted comparison table to the terminal."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON")
    print("=" * 100)

    header = (
        f"{'Model':<18} {'Type':<10} {'QA R-L':>8} {'QA BERT':>8} "
        f"{'CoT R-L':>8} {'CoT BERT':>9} {'Think%':>7} "
        f"{'Halluc%':>8} {'Overall':>8}"
    )
    print(header)
    print("-" * 100)

    for model_name, result in benchmark_data["models"].items():
        for eval_type, label in [("base_eval", "base"),
                                  ("finetuned_eval", "finetuned")]:
            metrics = result.get(eval_type)
            if metrics is None:
                continue
            print(
                f"{model_name:<18} {label:<10} "
                f"{metrics.get('qa_rougeL', 0):>8.4f} "
                f"{metrics.get('qa_bertscore_f1', 0):>8.4f} "
                f"{metrics.get('cot_rougeL', 0):>8.4f} "
                f"{metrics.get('cot_bertscore_f1', 0):>9.4f} "
                f"{metrics.get('cot_think_tag_rate', 0):>7.1%} "
                f"{metrics.get('hallucination_pass_rate', 0):>8.1%} "
                f"{metrics.get('overall_weighted_score', 0):>8.4f}"
            )

    print("=" * 100)

    # Training summary
    print(f"\n{'Model':<18} {'Loss':>8} {'Time':>12} "
          f"{'Trainable':>14} {'Trainable%':>11}")
    print("-" * 70)
    for model_name, result in benchmark_data["models"].items():
        t = result.get("training")
        if t is None:
            continue
        wall = t.get("wall_time_seconds", 0)
        hours = int(wall // 3600)
        minutes = int((wall % 3600) // 60)
        print(
            f"{model_name:<18} "
            f"{t.get('final_train_loss', 0) or 0:>8.4f} "
            f"{hours:>4}h {minutes:>2}m      "
            f"{t.get('trainable_params', 0):>14,} "
            f"{t.get('trainable_pct', 0):>10.2f}%"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Save LoRA adapters
# ---------------------------------------------------------------------------

def save_finetuned_model(model, tokenizer, model_name, base_dir):
    """Save both LoRA adapters and merged bf16 model for a fine-tuned model.

    Saves to:
        base_dir/lora_adapters/<model_name>/   — small LoRA adapter weights
        base_dir/merged_models/<model_name>/   — full merged bf16 model
    """
    # LoRA adapters (small, fast to save)
    lora_dir = os.path.join(base_dir, "lora_adapters", model_name)
    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"[save] LoRA adapters saved to {lora_dir}")

    # Merged bf16 model (larger, for direct inference without adapter loading)
    merged_dir = os.path.join(base_dir, "merged_models", model_name)
    os.makedirs(merged_dir, exist_ok=True)
    try:
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"[save] Merged bf16 model saved to {merged_dir}")
    except Exception as e:
        print(f"[save] WARNING: Could not save merged model for {model_name}: {e}")

    return lora_dir


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args):
    """Run the full benchmark pipeline.

    For each model:
      1. Load base model -> evaluate (base scores)
      2. Reload model -> apply LoRA -> train -> evaluate (fine-tuned scores)
      3. Save results locally
    Finally: aggregate results, log WandB comparison, print table.
    """
    from model_registry import (
        get_model_config, validate_model_names, resolve_model_path,
    )

    model_names = validate_model_names(args.models)

    print(f"[benchmark] Models to benchmark: {model_names}")
    print(f"[benchmark] Epochs: {args.epochs}, LR: {args.learning_rate}, "
          f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    # Ensure processed dataset exists
    ensure_processed_data(args.data_dir, args.processed_data_dir)

    # Benchmark metadata
    benchmark_id = f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb_group = benchmark_id
    benchmark_data = {
        "benchmark_id": benchmark_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_length": args.max_seq_length,
            "effective_batch_size": 16,
            "data_dir": args.data_dir,
            "seed": args.seed,
        },
        "models": {},
    }

    checkpoint_base = os.path.join(args.results_dir, benchmark_id, "checkpoints")

    # ======================================================================
    # Main loop: process each model sequentially
    # ======================================================================
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"  BENCHMARKING: {model_name}")
        print(f"{'='*70}")

        config = get_model_config(model_name)
        model_path = resolve_model_path(model_name, args.local_model_dir)
        model_result = {
            "model_name": model_name,
            "model_id": config["model_id"],
            "params_b": config["params_b"],
            "base_eval": None,
            "finetuned_eval": None,
            "training": None,
            "error": None,
        }

        try:
            # ----------------------------------------------------------
            # Phase 1: Base model evaluation
            # ----------------------------------------------------------
            if not args.skip_base_eval:
                print(f"\n[{model_name}] Phase 1: Base model evaluation ...")
                model, tokenizer = load_model(
                    model_path, args.max_seq_length, args.local_model_dir
                )
                base_summary = evaluate_model(
                    model, tokenizer, args.data_dir
                )
                model_result["base_eval"] = flatten_eval_results(base_summary)
                print(f"[{model_name}] Base overall: "
                      f"{model_result['base_eval']['overall_weighted_score']:.4f}")

                # Free GPU memory before training
                del model
                cleanup_gpu()

            # ----------------------------------------------------------
            # Phase 2: LoRA training + fine-tuned evaluation
            # ----------------------------------------------------------
            print(f"\n[{model_name}] Phase 2: LoRA training ...")
            model, tokenizer = load_model(
                model_path, args.max_seq_length, args.local_model_dir
            )
            model = apply_lora(
                model, args.lora_r, args.lora_alpha, args.max_seq_length
            )

            # Load and format dataset with this model's tokenizer
            dataset = load_and_format_dataset(
                args.processed_data_dir, tokenizer
            )

            # Train
            training_metrics = train_model(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                model_name=model_name,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                max_seq_length=args.max_seq_length,
                batch_size=config["batch_size"],
                gradient_accumulation_steps=config[
                    "gradient_accumulation_steps"
                ],
                output_dir=checkpoint_base,
                wandb_project=args.wandb_project,
                wandb_group=wandb_group,
                seed=args.seed,
            )
            model_result["training"] = training_metrics

            # Evaluate fine-tuned model
            print(f"\n[{model_name}] Phase 3: Fine-tuned evaluation ...")
            ft_summary = evaluate_model(model, tokenizer, args.data_dir)
            model_result["finetuned_eval"] = flatten_eval_results(ft_summary)
            print(f"[{model_name}] Fine-tuned overall: "
                  f"{model_result['finetuned_eval']['overall_weighted_score']:.4f}")

            # Save LoRA adapters + merged model
            save_base = os.path.join(args.results_dir, benchmark_id)
            save_finetuned_model(model, tokenizer, model_name, save_base)

        except Exception as e:
            print(f"\n[ERROR] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            model_result["error"] = str(e)

        finally:
            # Cleanup GPU memory regardless of success/failure
            for var_name in ["model", "tokenizer", "dataset"]:
                if var_name in dir():
                    try:
                        exec(f"del {var_name}")
                    except Exception:
                        pass
            cleanup_gpu()

        # Save per-model results
        benchmark_data["models"][model_name] = model_result
        per_model_dir = os.path.join(args.results_dir, benchmark_id, "per_model")
        save_model_result(per_model_dir, model_name, model_result)

    # ======================================================================
    # Aggregate results
    # ======================================================================
    save_benchmark_results(
        os.path.join(args.results_dir, benchmark_id), benchmark_data
    )

    # WandB comparison table
    try:
        wandb_url = log_comparison_to_wandb(
            benchmark_data, args.wandb_project, wandb_group
        )
        if wandb_url:
            print(f"\n[wandb] Comparison dashboard: {wandb_url}")
    except Exception as e:
        print(f"[wandb] Failed to log comparison: {e}")

    # Print terminal comparison
    print_comparison_table(benchmark_data)

    print(f"\n[benchmark] Results saved to: "
          f"{os.path.abspath(os.path.join(args.results_dir, benchmark_id))}")

    return benchmark_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    main()
