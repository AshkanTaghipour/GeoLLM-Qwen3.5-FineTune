"""
train.py — LoRA fine-tuning of Qwen 3.5-0.8B using Unsloth + SFTTrainer.

Key design decisions for someone coming from video-gen model training:
  - We use bf16 LoRA, NOT QLoRA (4-bit quantized LoRA). Unsloth's docs explicitly
    say QLoRA is not recommended for Qwen 3.5. The 0.8B model is small enough that
    full bf16 LoRA only needs ~3 GB VRAM — trivial on an A100-80GB.
  - LoRA rank r=16 with alpha=16 gives an effective scaling factor of 1.0
    (alpha/r = 1). This is a common conservative starting point. You can try
    r=32 or r=64 later if you want more capacity.
  - We use Unsloth's "unsloth" gradient checkpointing mode, which is faster than
    HuggingFace's default gradient checkpointing. Gradient checkpointing trades
    compute for memory: instead of storing all intermediate activations for the
    backward pass, it recomputes them on the fly. Saves ~60% memory at ~20% speed
    cost. On an A100 with a tiny 0.8B model you don't strictly need it, but it
    lets you push batch size or sequence length higher.
  - We use adamw_8bit optimizer to save a bit of memory on optimizer states.
    For a 0.8B model it barely matters, but it's good practice for when you
    scale up.

Usage:
    python train.py
    python train.py --epochs 5 --batch_size 8 --learning_rate 1e-4
    python train.py --model_name unsloth/Qwen3.5-0.8B --data_dir ./processed_data
"""

import argparse
import os
import sys
import time
import math

import torch
import wandb
from datasets import load_from_disk
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen 3.5-0.8B with Unsloth."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./models/Qwen3.5-0.8B",
        help="Path to local model or HuggingFace model ID. "
             "Pre-downloaded to ./models/Qwen3.5-0.8B by default.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./processed_data",
        help="Path to the HuggingFace Dataset saved by prepare_data.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Checkpoint directory used during training.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank. Higher = more trainable params, more capacity.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA scaling factor. Effective scale = alpha / r.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of full passes over the training set.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device train batch size. A100-80GB handles 4 easily for 0.8B.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Peak learning rate (after warmup).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum token sequence length. Longer = more memory.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Accumulate gradients over N mini-batches before updating. "
             "Effective batch = batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        default="./models",
        help="Local cache directory for downloaded model weights.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Custom metrics callback
# ---------------------------------------------------------------------------

class DetailedMetricsCallback(TrainerCallback):
    """
    Logs extra metrics to WandB that the default SFTTrainer does not:
      - GPU memory usage (allocated vs reserved)
      - Tokens processed per second (throughput)

    The Trainer already logs train/grad_norm (= gradient L2 norm) automatically.
    No need to duplicate it here — just check train/grad_norm in WandB.

    All metrics are logged to WandB so you can inspect them in real-time
    from any browser at https://wandb.ai.
    """

    def __init__(self):
        super().__init__()
        self.step_start_time = None

    # Called right before the training step (forward + backward).
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    # Called when the Trainer logs metrics (after each logging_steps).
    # We log GPU memory and throughput here. Note: the Trainer already logs
    # train/grad_norm which IS the L2 norm of all gradients (computed before
    # clipping/zero_grad). No need to duplicate it — just look at
    # train/grad_norm in your WandB dashboard.
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or wandb.run is None:
            return

        metrics = {}

        # ----- GPU memory -----
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            metrics["system/gpu_mem_allocated_gb"] = allocated_gb
            metrics["system/gpu_mem_reserved_gb"] = reserved_gb

        # ----- Tokens per second -----
        if self.step_start_time is not None:
            elapsed = time.time() - self.step_start_time
            tokens_this_step = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * args.max_seq_length
            )
            tokens_per_sec = tokens_this_step / max(elapsed, 1e-6)
            metrics["throughput/tokens_per_sec"] = tokens_per_sec

        if metrics:
            wandb.log(metrics)


# ---------------------------------------------------------------------------
# Chat-template formatting function
# ---------------------------------------------------------------------------

def make_formatting_func(tokenizer):
    """
    Return a function that takes a batch of examples (each with a "messages"
    column) and applies the tokenizer's chat template to produce a single
    string per example.

    The "messages" column comes from prepare_data.py and looks like:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."},
        ]

    SFTTrainer will then tokenize these strings internally.
    """
    def formatting_func(examples):
        texts = []
        for conversation in examples["messages"]:
            # apply_chat_template converts the list-of-dicts into the model's
            # expected prompt format (with special tokens, role tags, etc.)
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    return formatting_func


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Set weight download location
    # ------------------------------------------------------------------
    # By default HuggingFace downloads models to ~/.cache/huggingface/ which
    # can fill up your home directory on shared clusters. We redirect to a
    # project-local folder instead.
    os.makedirs(args.local_model_dir, exist_ok=True)
    abs_model_dir = os.path.abspath(args.local_model_dir)
    os.environ["HF_HOME"] = abs_model_dir
    os.environ["TRANSFORMERS_CACHE"] = abs_model_dir
    print(f"[config] Model cache directory: {abs_model_dir}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    # IMPORTANT: Use FastLanguageModel, NOT FastModel.
    # FastModel is for a different Unsloth pathway. For Qwen 3.5 text
    # models you must use FastLanguageModel.
    #
    # load_in_4bit=False: We intentionally avoid 4-bit (QLoRA). Unsloth's
    #   own documentation warns that QLoRA is NOT recommended for Qwen 3.5
    #   because of numerical instability in the quantised attention layers.
    # load_in_16bit=True: bf16 LoRA. The base weights stay in bf16 and only
    #   the small LoRA adapter matrices are trained. This is the sweet spot
    #   for a tiny 0.8B model on an A100 (~3 GB VRAM).
    # full_finetuning=False: We only train the LoRA adapters, not the full
    #   model weights. This is much faster and uses far less memory.

    from unsloth import FastLanguageModel  # import here so env vars are set first

    print(f"[model] Loading {args.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,       # NOT QLoRA — see note above
        load_in_16bit=True,       # bf16 LoRA
        full_finetuning=False,    # only train LoRA adapters
    )
    print("[model] Base model loaded.")

    # ------------------------------------------------------------------
    # 3. Apply LoRA adapters
    # ------------------------------------------------------------------
    # target_modules: We attach LoRA adapters to all the main linear layers
    #   in each transformer block. This is standard practice. Skipping some
    #   (e.g., only q/v) saves a tiny bit of memory but usually hurts quality.
    #
    # r=16, lora_alpha=16: The effective LoRA scaling is alpha/r = 1.0.
    #   Think of r as the "rank" of the low-rank update matrices — higher
    #   means more expressive but more parameters. 16 is a solid default.
    #
    # lora_dropout=0: Dropout on the LoRA layers. 0 is fine for most
    #   fine-tuning; you'd increase it if you saw overfitting.
    #
    # use_gradient_checkpointing="unsloth": Unsloth has a custom, faster
    #   implementation of gradient checkpointing. Standard gradient
    #   checkpointing (PyTorch's) recomputes activations during the backward
    #   pass to save memory. Unsloth's version does the same thing but with
    #   less overhead. On an A100 with a 0.8B model you have plenty of VRAM,
    #   but this lets you use larger batch sizes or longer sequences.

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention projections
            "gate_proj", "up_proj", "down_proj",       # MLP projections
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # faster than HF's default
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    # Quick summary of trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 4. Load and format the dataset
    # ------------------------------------------------------------------
    # The dataset was saved by prepare_data.py as a HuggingFace DatasetDict
    # with "train" and "test" splits. Each row has a "messages" column
    # containing a list of chat-format message dicts.

    print(f"[data] Loading dataset from {args.data_dir} ...")
    dataset = load_from_disk(args.data_dir)
    print(f"[data] Train size: {len(dataset['train']):,}  |  "
          f"Test size: {len(dataset['test']):,}")

    # Apply the chat template to convert messages → single text strings.
    # SFTTrainer will tokenize these strings for us.
    formatting_func = make_formatting_func(tokenizer)
    dataset = dataset.map(
        formatting_func,
        batched=True,
        desc="Applying chat template",
    )

    # ------------------------------------------------------------------
    # 5. Configure the trainer
    # ------------------------------------------------------------------
    # SFTTrainer (Supervised Fine-Tuning Trainer) from the `trl` library is
    # a thin wrapper around HuggingFace's Trainer that handles chat/instruct
    # data formats. Under the hood it does the same thing as the Trainer you
    # might know from image/video model training.
    #
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
    #   = 4 * 4 = 16 by default. This means we update weights every 16 examples.
    #
    # warmup_ratio=0.1: Linearly ramp up LR for the first 10% of training
    #   steps. Prevents early instability.
    #
    # lr_scheduler_type="cosine": After warmup, decay LR following a cosine
    #   curve down to ~0. This is the most common schedule for fine-tuning.
    #
    # report_to="wandb": Logs loss, eval metrics, etc. to WandB for real-time
    #   monitoring at https://wandb.ai. Our custom callback adds gradient norms,
    #   GPU memory, and throughput on top of the defaults.

    from trl import SFTTrainer, SFTConfig

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize WandB run — you'll see a URL printed that links to the dashboard
    wandb.init(
        project="qwen35-geology-finetune",
        name=f"lora-r{args.lora_r}-lr{args.learning_rate}-ep{args.epochs}",
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": "adamw_8bit",
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.1,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
        },
    )

    training_args = SFTConfig(
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        logging_steps=1,                 # log every single step for visibility
        eval_strategy="epoch",             # eval loss once per epoch (3 points total)
        save_strategy="steps",
        save_steps=50,                   # save checkpoint every 50 steps
        output_dir=args.output_dir,
        optim="adamw_8bit",              # 8-bit Adam — saves optimizer memory
        seed=3407,
        bf16=True,                       # match the model's dtype
        lr_scheduler_type="cosine",
        learning_rate=args.learning_rate,
        report_to="wandb",              # send default metrics to WandB
    )

    # Create the custom metrics callback (adds grad norm, GPU mem, throughput)
    metrics_callback = DetailedMetricsCallback()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[metrics_callback],
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training")
    print(f"  Model:          {args.model_name}")
    print(f"  LoRA rank:      {args.lora_r}  (alpha={args.lora_alpha})")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.gradient_accumulation_steps} "
          f"(effective {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Output dir:     {args.output_dir}")
    print("=" * 60 + "\n")

    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start

    # ------------------------------------------------------------------
    # 7. Save LoRA adapters
    # ------------------------------------------------------------------
    lora_save_dir = "./finetuned_lora"
    print(f"\n[save] Saving LoRA adapters to {lora_save_dir} ...")
    model.save_pretrained(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    print("[save] LoRA adapters saved.")

    # ------------------------------------------------------------------
    # 8. Save merged model (bf16)
    # ------------------------------------------------------------------
    # Merging folds the LoRA weights back into the base model so you get a
    # single set of weights for inference — no adapter loading needed.
    # This is convenient for deployment but takes more disk space.
    merged_save_dir = "./finetuned_merged"
    print(f"[save] Saving merged bf16 model to {merged_save_dir} ...")
    model.save_pretrained_merged(
        merged_save_dir,
        tokenizer,
        save_method="merged_16bit",  # bf16 merged weights
    )
    print("[save] Merged model saved.")

    # ------------------------------------------------------------------
    # 9. Training summary
    # ------------------------------------------------------------------
    total_steps = state_or_default(train_result, "global_step", trainer.state.global_step)
    final_loss = state_or_default(train_result, "training_loss", None)

    hours = int(train_elapsed // 3600)
    minutes = int((train_elapsed % 3600) // 60)
    seconds = int(train_elapsed % 60)

    # The HF Trainer with report_to="wandb" closes the WandB run internally
    # after training ends, so wandb.run may already be None here. We guard
    # against that — the metrics were already streamed during training anyway.
    wandb_url = None
    if wandb.run is not None:
        wandb_url = wandb.run.get_url()
        wandb.run.summary["wall_time_seconds"] = train_elapsed
        if final_loss is not None:
            wandb.run.summary["final_training_loss"] = final_loss
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Total steps:    {total_steps}")
    if final_loss is not None:
        print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Wall time:      {hours}h {minutes}m {seconds}s")
    print(f"  LoRA adapters:  {os.path.abspath(lora_save_dir)}")
    print(f"  Merged model:   {os.path.abspath(merged_save_dir)}")
    if wandb_url:
        print(f"  WandB dashboard: {wandb_url}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 10. Run evaluation on test sets
    # ------------------------------------------------------------------
    # Pass the already-loaded model/tokenizer so we don't reload from disk.
    print("\n[eval] Running post-training evaluation ...")
    from evaluate import run_evaluation
    run_evaluation(
        model_dir=lora_save_dir,
        base_model=args.model_name,
        model=model,
        tokenizer=tokenizer,
    )


def state_or_default(train_result, key, default):
    """Safely extract a field from the TrainOutput object."""
    if hasattr(train_result, key):
        return getattr(train_result, key)
    if isinstance(train_result, dict) and key in train_result:
        return train_result[key]
    # TrainOutput stores metrics in a dict
    if hasattr(train_result, "metrics") and key in train_result.metrics:
        return train_result.metrics[key]
    return default


if __name__ == "__main__":
    main()
