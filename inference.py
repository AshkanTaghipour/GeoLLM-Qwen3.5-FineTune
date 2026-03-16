"""
inference.py — Standalone inference script for GeoLLM fine-tuned models.

Downloads a model from HuggingFace and generates responses to geology questions.

Usage:
    python inference.py --question "What geophysical methods target komatiite nickel in the Eastern Goldfields?"
    python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --question "..."
    python inference.py --interactive
"""

import argparse

SYSTEM_PROMPT = (
    "You are a specialist geologist and exploration consultant with over "
    "10 years of experience in Western Australian and Queensland mineral "
    "exploration. You provide expert advice on geological interpretation, "
    "exploration methods, deposit models, geochemistry, geophysics, and "
    "drilling strategies. You answer like a knowledgeable colleague — concise, "
    "technically specific, and grounded in real geological data."
)

AVAILABLE_MODELS = [
    "AshkanTaghipour/GeoLLM-Qwen3.5-0.8B",
    "AshkanTaghipour/GeoLLM-Qwen3.5-2B",
    "AshkanTaghipour/GeoLLM-Qwen3.5-4B",
    "AshkanTaghipour/GeoLLM-Qwen3.5-9B",
    "AshkanTaghipour/GeoLLM-Qwen3.5-27B",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a GeoLLM fine-tuned model."
    )
    p.add_argument(
        "--model", default="AshkanTaghipour/GeoLLM-Qwen3.5-4B",
        help=f"HuggingFace model ID. Available: {', '.join(AVAILABLE_MODELS)}",
    )
    p.add_argument(
        "--question", default=None,
        help="Geology question to ask the model.",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate.",
    )
    p.add_argument(
        "--enable_thinking", action="store_true",
        help="Enable chain-of-thought reasoning (<think> tags).",
    )
    p.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode (ask multiple questions).",
    )
    return p.parse_args()


def load_model(model_name):
    """Load model and tokenizer from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}.")
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens=512,
             enable_thinking=False):
    """Generate a response for a geology question."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

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

    response = tokenizer.decode(
        outputs[0][prompt_len:], skip_special_tokens=True
    )
    return response.strip()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            response = generate(
                model, tokenizer, question,
                args.max_new_tokens, args.enable_thinking,
            )
            print(f"\nResponse:\n{response}\n")
    elif args.question:
        response = generate(
            model, tokenizer, args.question,
            args.max_new_tokens, args.enable_thinking,
        )
        print(f"\n{response}")
    else:
        print("Provide --question or --interactive. See --help.")


if __name__ == "__main__":
    main()
