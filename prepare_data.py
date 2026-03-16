"""
prepare_data.py

Transforms raw QA and chain-of-thought (CoT) training data into the chat
message format expected by Qwen 3.5-0.8B fine-tuning, and saves it as a
HuggingFace Dataset.

Supports two input formats:
  1. Pre-split: separate {qa,cot}_{train,test}.jsonl files (v1 layout)
  2. Combined:  single qa_pairs.jsonl + cot_pairs.jsonl files (v2 layout)
     In combined mode the script performs a stratified train/test split.

Usage:
    # v2 combined format (default) — splits data automatically
    python prepare_data.py --input_dir ./training_splits_v2 --output_dir ./processed_data

    # v1 pre-split format — uses existing splits
    python prepare_data.py --input_dir ./training_splits --output_dir ./processed_data

    # Custom split ratio and seed
    python prepare_data.py --test_ratio 0.10 --seed 123
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

# ---------------------------------------------------------------------------
# System prompt -- defines the persona the fine-tuned model should adopt.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a specialist geologist and exploration consultant with over "
    "10 years of experience in Western Australian and Queensland mineral "
    "exploration. You provide expert advice on geological interpretation, "
    "exploration methods, deposit models, geochemistry, geophysics, and "
    "drilling strategies. You answer like a knowledgeable colleague — concise, "
    "technically specific, and grounded in real geological data."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(filepath: str) -> List[Dict]:
    """Load a JSONL file, returning a list of dicts (one per line)."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(
    records: List[Dict], filepath: str, keys: Optional[List[str]] = None
) -> None:
    """Save records to a JSONL file.  If *keys* is given, only those keys
    are written (strips metadata like batch_id, source_anumbers)."""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            if keys:
                record = {k: record[k] for k in keys if k in record}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Input format detection
# ---------------------------------------------------------------------------

def detect_input_format(input_dir: str) -> str:
    """Return 'combined' or 'pre-split' based on the files present."""
    p = Path(input_dir)
    has_combined = (
        (p / "qa_pairs.jsonl").exists() and (p / "cot_pairs.jsonl").exists()
    )
    has_presplit = (
        (p / "qa_train.jsonl").exists()
        and (p / "qa_test.jsonl").exists()
        and (p / "cot_train.jsonl").exists()
        and (p / "cot_test.jsonl").exists()
    )
    if has_combined:
        return "combined"
    if has_presplit:
        return "pre-split"
    raise FileNotFoundError(
        f"Cannot detect input format in {input_dir}. "
        "Expected either qa_pairs.jsonl + cot_pairs.jsonl (combined) "
        "or qa_train/test.jsonl + cot_train/test.jsonl (pre-split)."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_records(
    records: List[Dict], data_type: str, filepath: str
) -> Tuple[List[Dict], List[Dict]]:
    """Validate records and return (valid, demoted_to_qa).

    QA records need non-empty question + answer.
    CoT records need non-empty question + reasoning + answer.
    CoT records missing *only* the reasoning field are demoted to QA
    (the question and answer are still useful for training).
    Records with other missing fields are dropped entirely.
    """
    valid: List[Dict] = []
    demoted_to_qa: List[Dict] = []
    required = ["question", "answer"]
    if data_type == "cot":
        required.append("reasoning")

    for i, rec in enumerate(records):
        missing = [k for k in required if not rec.get(k, "").strip()]
        if not missing:
            valid.append(rec)
        elif data_type == "cot" and missing == ["reasoning"]:
            print(
                f"  WARNING: {filepath} record {i}: missing 'reasoning', "
                f"demoting to QA. Q: {rec['question'][:80]}..."
            )
            demoted_to_qa.append(rec)
        else:
            print(
                f"  SKIPPED: {filepath} record {i}: missing {missing}. "
                f"Q: {rec.get('question', 'N/A')[:80]}..."
            )
    return valid, demoted_to_qa


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    qa_records: List[Dict],
    cot_records: List[Dict],
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Split QA and CoT separately at the same ratio (stratified split).

    Stratifying guarantees both data types are proportionally represented
    in train and test, which is important when the test set is small.
    Returns (qa_train, qa_test, cot_train, cot_test).
    """
    rng = random.Random(seed)

    qa = list(qa_records)
    cot = list(cot_records)
    rng.shuffle(qa)
    rng.shuffle(cot)

    qa_n_test = max(1, round(len(qa) * test_ratio))
    cot_n_test = max(1, round(len(cot) * test_ratio))

    return (
        qa[qa_n_test:],   # qa_train
        qa[:qa_n_test],   # qa_test
        cot[cot_n_test:], # cot_train
        cot[:cot_n_test], # cot_test
    )


# ---------------------------------------------------------------------------
# Chat-format conversion
# ---------------------------------------------------------------------------

def format_qa_example(entry: Dict) -> Dict:
    """Convert a plain QA pair into chat message format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry["question"]},
        {"role": "assistant", "content": entry["answer"]},
    ]
    return {"messages": messages, "data_type": "qa"}


def format_cot_example(entry: Dict) -> Dict:
    """Convert a CoT entry into chat message format with <think> tags."""
    assistant_content = (
        f"<think>\n{entry['reasoning']}\n</think>\n\n{entry['answer']}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry["question"]},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"messages": messages, "data_type": "cot"}


def combine_and_format(
    qa_records: List[Dict],
    cot_records: List[Dict],
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """Convert QA + CoT records to chat format, combine, optionally shuffle."""
    examples = [format_qa_example(e) for e in qa_records]
    examples += [format_cot_example(e) for e in cot_records]
    if shuffle:
        random.seed(seed)
        random.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(examples: List[Dict], split_name: str) -> None:
    """Print summary statistics for a processed split."""
    qa_count = sum(1 for ex in examples if ex["data_type"] == "qa")
    cot_count = sum(1 for ex in examples if ex["data_type"] == "cot")
    total = len(examples)

    total_chars = sum(
        sum(len(m["content"]) for m in ex["messages"]) for ex in examples
    )
    avg_len = total_chars / total if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  {split_name.upper()} split statistics")
    print(f"{'=' * 50}")
    print(f"  QA examples:    {qa_count}")
    print(f"  CoT examples:   {cot_count}")
    print(f"  Total examples: {total}")
    print(f"  Avg message length (chars): {avg_len:,.0f}")
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    input_dir: str,
    output_dir: str,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """Load raw data, validate, split if needed, convert to chat format,
    build a HuggingFace DatasetDict, and save to disk.

    When the input is combined format (v2), split JSONL files are also
    saved to the input directory for use by evaluate.py.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    fmt = detect_input_format(input_dir)
    print(f"  Detected input format: {fmt}")

    if fmt == "combined":
        # ----- Load combined files -----
        print("\n  Loading combined data files...")
        qa_all = load_jsonl(str(input_path / "qa_pairs.jsonl"))
        cot_all = load_jsonl(str(input_path / "cot_pairs.jsonl"))
        print(f"  Loaded {len(qa_all)} QA pairs, {len(cot_all)} CoT pairs")

        # ----- Validate -----
        print("\n  Validating records...")
        qa_valid, _ = validate_records(qa_all, "qa", "qa_pairs.jsonl")
        cot_valid, demoted = validate_records(cot_all, "cot", "cot_pairs.jsonl")
        if demoted:
            print(f"  -> {len(demoted)} CoT records demoted to QA (missing reasoning)")
            qa_valid.extend(demoted)
        print(f"  After validation: {len(qa_valid)} QA, {len(cot_valid)} CoT")

        # ----- Stratified split -----
        print(
            f"\n  Stratified split: {1 - test_ratio:.0%} train / "
            f"{test_ratio:.0%} test  (seed={seed})"
        )
        qa_train, qa_test, cot_train, cot_test = stratified_split(
            qa_valid, cot_valid, test_ratio=test_ratio, seed=seed
        )
        print(
            f"  Train: {len(qa_train)} QA + {len(cot_train)} CoT = "
            f"{len(qa_train) + len(cot_train)}"
        )
        print(
            f"  Test:  {len(qa_test)} QA + {len(cot_test)} CoT = "
            f"{len(qa_test) + len(cot_test)}"
        )

        # ----- Save split JSONL files for evaluate.py -----
        save_jsonl(qa_train, str(input_path / "qa_train.jsonl"),
                   keys=["question", "answer"])
        save_jsonl(qa_test, str(input_path / "qa_test.jsonl"),
                   keys=["question", "answer"])
        save_jsonl(cot_train, str(input_path / "cot_train.jsonl"),
                   keys=["question", "reasoning", "answer"])
        save_jsonl(cot_test, str(input_path / "cot_test.jsonl"),
                   keys=["question", "reasoning", "answer"])
        print(f"  Split JSONL files saved to {input_path}/")

    else:
        # ----- Pre-split format: load directly -----
        qa_train = load_jsonl(str(input_path / "qa_train.jsonl"))
        qa_test = load_jsonl(str(input_path / "qa_test.jsonl"))
        cot_train = load_jsonl(str(input_path / "cot_train.jsonl"))
        cot_test = load_jsonl(str(input_path / "cot_test.jsonl"))

    # ----- Format into chat messages -----
    train_examples = combine_and_format(
        qa_train, cot_train, shuffle=True, seed=seed
    )
    test_examples = combine_and_format(qa_test, cot_test, shuffle=False)

    compute_statistics(train_examples, "train")
    compute_statistics(test_examples, "test")

    # ----- Build HuggingFace DatasetDict -----
    train_dataset = Dataset.from_dict({
        "messages": [ex["messages"] for ex in train_examples],
        "data_type": [ex["data_type"] for ex in train_examples],
    })
    test_dataset = Dataset.from_dict({
        "messages": [ex["messages"] for ex in test_examples],
        "data_type": [ex["data_type"] for ex in test_examples],
    })
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    print(f"\nHuggingFace Dataset saved to: {output_path.resolve()}")

    # ----- Show a sample -----
    print("\n--- Sample train example (first entry) ---")
    sample = train_dataset[0]
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]
        preview = content[:120] + "..." if len(content) > 120 else content
        print(f"  [{role}] {preview}")
    print(f"  data_type: {sample['data_type']}")

    return dataset_dict


# ---------------------------------------------------------------------------
# Convenience wrappers (used by tests and other modules)
# ---------------------------------------------------------------------------

def load_qa_data(filepath: str = None) -> List[Dict]:
    """Load QA training data."""
    if filepath is None:
        filepath = str(Path(__file__).parent / "training_splits_v2" / "qa_train.jsonl")
    return load_jsonl(filepath)


def load_cot_data(filepath: str = None) -> List[Dict]:
    """Load CoT training data."""
    if filepath is None:
        filepath = str(Path(__file__).parent / "training_splits_v2" / "cot_train.jsonl")
    return load_jsonl(filepath)


def create_dataset(
    input_dir: str = None, output_dir: str = None
) -> DatasetDict:
    """Build and return the dataset."""
    if input_dir is None:
        input_dir = str(Path(__file__).parent / "training_splits_v2")
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "processed_data")
    return build_dataset(input_dir, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw geology QA/CoT data for Qwen 3.5-0.8B fine-tuning. "
            "Converts JSONL files into HuggingFace Dataset with chat message "
            "format.  Auto-detects whether the input directory uses combined "
            "files (v2) or pre-split files (v1)."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./training_splits_v2",
        help="Directory containing data files (default: ./training_splits_v2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save the HuggingFace Dataset (default: ./processed_data)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Fraction of data for the test split (default: 0.05). "
             "Only used with combined-format input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and shuffling (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Preparing fine-tuning data for Qwen 3.5-0.8B with thinking support...")
    print(f"  Input:  {os.path.abspath(args.input_dir)}")
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    build_dataset(
        args.input_dir,
        args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print("\nDone! You can load this dataset with:")
    print("  from datasets import load_from_disk")
    print(f'  ds = load_from_disk("{args.output_dir}")')
