"""
Comprehensive tests for the Qwen 3.5-0.8B fine-tuning data preparation and training pipeline.

Run all non-GPU tests:
    pytest test_training.py -m "not gpu" -v

Run everything (requires GPU node):
    pytest test_training.py -v
"""

import json
import os
import random
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_splits")

QA_TRAIN_PATH = os.path.join(DATA_DIR, "qa_train.jsonl")
COT_TRAIN_PATH = os.path.join(DATA_DIR, "cot_train.jsonl")

SYSTEM_PROMPT = (
    "You are a specialist geologist and exploration consultant with over 10 years "
    "of experience in Western Australian and Queensland mineral exploration. You "
    "provide expert advice on geological interpretation, exploration methods, "
    "deposit models, geochemistry, geophysics, and drilling strategies. You answer "
    "like a knowledgeable colleague \u2014 concise, technically specific, and grounded "
    "in real geological data."
)

# Expected dataset sizes
EXPECTED_QA_COUNT = 88
EXPECTED_COT_COUNT = 60
EXPECTED_TOTAL_COUNT = EXPECTED_QA_COUNT + EXPECTED_COT_COUNT  # 148


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def qa_data():
    """Load all QA training examples from qa_train.jsonl."""
    records = []
    with open(QA_TRAIN_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


@pytest.fixture
def cot_data():
    """Load all CoT training examples from cot_train.jsonl."""
    records = []
    with open(COT_TRAIN_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


@pytest.fixture
def prepare_data_module():
    """Import the prepare_data module, skipping if unavailable."""
    try:
        import prepare_data
        return prepare_data
    except ImportError:
        pytest.skip("prepare_data module not yet available")


# ===========================================================================
# 1. Data Preparation Tests  (no GPU required)
# ===========================================================================

class TestDataLoading:
    """Tests that raw JSONL files load correctly with expected structure."""

    def test_qa_train_file_exists(self):
        """qa_train.jsonl must exist on disk."""
        assert os.path.isfile(QA_TRAIN_PATH), f"Missing {QA_TRAIN_PATH}"

    def test_cot_train_file_exists(self):
        """cot_train.jsonl must exist on disk."""
        assert os.path.isfile(COT_TRAIN_PATH), f"Missing {COT_TRAIN_PATH}"

    def test_qa_data_has_expected_keys(self, qa_data):
        """Every QA record must contain 'question' and 'answer' keys."""
        for i, record in enumerate(qa_data):
            assert "question" in record, f"QA record {i} missing 'question'"
            assert "answer" in record, f"QA record {i} missing 'answer'"

    def test_cot_data_has_expected_keys(self, cot_data):
        """Every CoT record must contain 'question', 'reasoning', and 'answer'."""
        for i, record in enumerate(cot_data):
            assert "question" in record, f"CoT record {i} missing 'question'"
            assert "reasoning" in record, f"CoT record {i} missing 'reasoning'"
            assert "answer" in record, f"CoT record {i} missing 'answer'"

    def test_qa_record_count(self, qa_data):
        """QA training set should have exactly 88 examples."""
        assert len(qa_data) == EXPECTED_QA_COUNT, (
            f"Expected {EXPECTED_QA_COUNT} QA records, got {len(qa_data)}"
        )

    def test_cot_record_count(self, cot_data):
        """CoT training set should have exactly 60 examples."""
        assert len(cot_data) == EXPECTED_COT_COUNT, (
            f"Expected {EXPECTED_COT_COUNT} CoT records, got {len(cot_data)}"
        )

    def test_combined_dataset_count(self, qa_data, cot_data):
        """Combined dataset should total 148 training examples."""
        total = len(qa_data) + len(cot_data)
        assert total == EXPECTED_TOTAL_COUNT, (
            f"Expected {EXPECTED_TOTAL_COUNT} total records, got {total}"
        )


class TestDataFormatting:
    """Tests that QA and CoT records are formatted into the correct chat structure."""

    @staticmethod
    def format_qa_example(record):
        """Format a QA record into the 3-message chat structure.

        This mirrors what prepare_data.format_qa_example() should produce.
        """
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": record["answer"]},
            ]
        }

    @staticmethod
    def format_cot_example(record):
        """Format a CoT record, wrapping reasoning in <think>...</think> tags.

        This mirrors what prepare_data.format_cot_example() should produce.
        """
        assistant_content = (
            f"<think>\n{record['reasoning']}\n</think>\n\n{record['answer']}"
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["question"]},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    def test_qa_format_produces_three_messages(self, qa_data):
        """QA formatting must produce exactly 3 messages (system, user, assistant)."""
        formatted = self.format_qa_example(qa_data[0])
        assert len(formatted["messages"]) == 3

    def test_qa_format_message_roles(self, qa_data):
        """QA messages must have roles: system, user, assistant (in order)."""
        formatted = self.format_qa_example(qa_data[0])
        roles = [m["role"] for m in formatted["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_qa_system_prompt_is_correct(self, qa_data):
        """QA system message must match the expected system prompt."""
        formatted = self.format_qa_example(qa_data[0])
        assert formatted["messages"][0]["content"] == SYSTEM_PROMPT

    def test_qa_assistant_has_no_think_tag(self, qa_data):
        """QA assistant content must NOT contain <think> tags."""
        formatted = self.format_qa_example(qa_data[0])
        assistant_content = formatted["messages"][2]["content"]
        assert "<think>" not in assistant_content
        assert "</think>" not in assistant_content

    def test_cot_format_produces_three_messages(self, cot_data):
        """CoT formatting must produce exactly 3 messages (system, user, assistant)."""
        formatted = self.format_cot_example(cot_data[0])
        assert len(formatted["messages"]) == 3

    def test_cot_format_message_roles(self, cot_data):
        """CoT messages must have roles: system, user, assistant (in order)."""
        formatted = self.format_cot_example(cot_data[0])
        roles = [m["role"] for m in formatted["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_cot_system_prompt_is_correct(self, cot_data):
        """CoT system message must match the expected system prompt."""
        formatted = self.format_cot_example(cot_data[0])
        assert formatted["messages"][0]["content"] == SYSTEM_PROMPT

    def test_cot_assistant_has_think_tags(self, cot_data):
        """CoT assistant content must wrap reasoning in <think>...</think> tags."""
        formatted = self.format_cot_example(cot_data[0])
        assistant_content = formatted["messages"][2]["content"]
        assert "<think>" in assistant_content
        assert "</think>" in assistant_content

    def test_cot_think_tag_contains_reasoning(self, cot_data):
        """The text between <think> tags must contain the original reasoning."""
        record = cot_data[0]
        formatted = self.format_cot_example(record)
        assistant_content = formatted["messages"][2]["content"]
        # Extract what's between the think tags
        start = assistant_content.index("<think>") + len("<think>\n")
        end = assistant_content.index("\n</think>")
        reasoning_in_tags = assistant_content[start:end]
        assert reasoning_in_tags == record["reasoning"]

    def test_cot_answer_follows_think_block(self, cot_data):
        """The final answer must appear after the </think> tag in CoT format."""
        record = cot_data[0]
        formatted = self.format_cot_example(record)
        assistant_content = formatted["messages"][2]["content"]
        think_end = assistant_content.index("</think>")
        after_think = assistant_content[think_end + len("</think>"):]
        assert record["answer"] in after_think

    def test_format_all_qa_examples(self, qa_data):
        """All QA records must format without error."""
        for i, record in enumerate(qa_data):
            formatted = self.format_qa_example(record)
            assert len(formatted["messages"]) == 3, f"QA record {i} format error"

    def test_format_all_cot_examples(self, cot_data):
        """All CoT records must format without error."""
        for i, record in enumerate(cot_data):
            formatted = self.format_cot_example(record)
            assert "<think>" in formatted["messages"][2]["content"], (
                f"CoT record {i} missing <think> tag"
            )


class TestDataShuffling:
    """Tests that data shuffling is deterministic with a fixed seed."""

    def test_shuffle_reproducibility_with_seed(self, qa_data, cot_data):
        """Shuffling the combined dataset with seed=42 must produce the same order."""
        combined_a = list(range(len(qa_data) + len(cot_data)))
        combined_b = list(range(len(qa_data) + len(cot_data)))

        rng_a = random.Random(42)
        rng_a.shuffle(combined_a)

        rng_b = random.Random(42)
        rng_b.shuffle(combined_b)

        assert combined_a == combined_b, "Same seed must produce identical ordering"

    def test_shuffle_differs_without_same_seed(self, qa_data, cot_data):
        """Shuffling with different seeds must (very likely) produce different orders."""
        combined_a = list(range(len(qa_data) + len(cot_data)))
        combined_b = list(range(len(qa_data) + len(cot_data)))

        rng_a = random.Random(42)
        rng_a.shuffle(combined_a)

        rng_b = random.Random(99)
        rng_b.shuffle(combined_b)

        assert combined_a != combined_b, "Different seeds should produce different orders"


# ===========================================================================
# 2. Data Quality Tests
# ===========================================================================

class TestDataQuality:
    """Tests that training data has no degenerate or malformed entries."""

    def test_no_empty_qa_questions(self, qa_data):
        """No QA record should have an empty or whitespace-only question."""
        for i, record in enumerate(qa_data):
            assert record["question"].strip(), f"QA record {i} has empty question"

    def test_no_empty_qa_answers(self, qa_data):
        """No QA record should have an empty or whitespace-only answer."""
        for i, record in enumerate(qa_data):
            assert record["answer"].strip(), f"QA record {i} has empty answer"

    def test_no_empty_cot_questions(self, cot_data):
        """No CoT record should have an empty or whitespace-only question."""
        for i, record in enumerate(cot_data):
            assert record["question"].strip(), f"CoT record {i} has empty question"

    def test_no_empty_cot_answers(self, cot_data):
        """No CoT record should have an empty or whitespace-only answer."""
        for i, record in enumerate(cot_data):
            assert record["answer"].strip(), f"CoT record {i} has empty answer"

    def test_no_empty_cot_reasoning(self, cot_data):
        """No CoT record should have an empty or whitespace-only reasoning field."""
        for i, record in enumerate(cot_data):
            assert record["reasoning"].strip(), f"CoT record {i} has empty reasoning"

    def test_qa_message_roles_valid(self, qa_data):
        """When formatted, all QA messages must use valid roles."""
        valid_roles = {"system", "user", "assistant"}
        for i, record in enumerate(qa_data):
            formatted = TestDataFormatting.format_qa_example(record)
            for msg in formatted["messages"]:
                assert msg["role"] in valid_roles, (
                    f"QA record {i} has invalid role '{msg['role']}'"
                )

    def test_cot_message_roles_valid(self, cot_data):
        """When formatted, all CoT messages must use valid roles."""
        valid_roles = {"system", "user", "assistant"}
        for i, record in enumerate(cot_data):
            formatted = TestDataFormatting.format_cot_example(record)
            for msg in formatted["messages"]:
                assert msg["role"] in valid_roles, (
                    f"CoT record {i} has invalid role '{msg['role']}'"
                )

    def test_qa_question_length_reasonable(self, qa_data):
        """QA questions should be between 10 and 5000 characters."""
        for i, record in enumerate(qa_data):
            length = len(record["question"])
            assert 10 <= length <= 5000, (
                f"QA record {i} question length {length} outside [10, 5000]"
            )

    def test_qa_answer_length_reasonable(self, qa_data):
        """QA answers should be between 50 and 10000 characters."""
        for i, record in enumerate(qa_data):
            length = len(record["answer"])
            assert 50 <= length <= 10000, (
                f"QA record {i} answer length {length} outside [50, 10000]"
            )

    def test_cot_question_length_reasonable(self, cot_data):
        """CoT questions should be between 10 and 5000 characters."""
        for i, record in enumerate(cot_data):
            length = len(record["question"])
            assert 10 <= length <= 5000, (
                f"CoT record {i} question length {length} outside [10, 5000]"
            )

    def test_cot_answer_length_reasonable(self, cot_data):
        """CoT answers should be between 50 and 10000 characters."""
        for i, record in enumerate(cot_data):
            length = len(record["answer"])
            assert 50 <= length <= 10000, (
                f"CoT record {i} answer length {length} outside [50, 10000]"
            )

    def test_cot_reasoning_length_reasonable(self, cot_data):
        """CoT reasoning should be between 50 and 15000 characters."""
        for i, record in enumerate(cot_data):
            length = len(record["reasoning"])
            assert 50 <= length <= 15000, (
                f"CoT record {i} reasoning length {length} outside [50, 15000]"
            )


# ===========================================================================
# 3. Training Config Tests  (no GPU required)
# ===========================================================================

class TestTrainingConfig:
    """Tests that training configuration objects can be constructed without error."""

    def test_sft_config_valid(self):
        """SFTConfig should construct with our intended training arguments."""
        try:
            from trl import SFTConfig
        except ImportError:
            pytest.skip("trl not installed")

        config = SFTConfig(
            output_dir=tempfile.mkdtemp(),
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=10,
            logging_steps=5,
            save_strategy="epoch",
            bf16=False,
            fp16=False,
            max_length=2048,  # trl 0.24+ uses max_length; unsloth patches max_seq_length
            seed=42,
        )
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 2
        assert config.learning_rate == 2e-4
        assert config.seed == 42

    def test_lora_config_parameters_reasonable(self):
        """LoRA config parameters must be positive and within sensible ranges."""
        try:
            from peft import LoraConfig
        except ImportError:
            pytest.skip("peft not installed")

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        assert config.r > 0, "LoRA rank must be positive"
        assert config.lora_alpha > 0, "LoRA alpha must be positive"
        assert 0.0 <= config.lora_dropout < 1.0, "Dropout must be in [0, 1)"
        assert len(config.target_modules) > 0, "Must target at least one module"

    def test_output_directory_can_be_created(self):
        """Output directories for checkpoints must be creatable."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "qwen_finetuned_output")
            os.makedirs(output_dir, exist_ok=True)
            assert os.path.isdir(output_dir)


# ===========================================================================
# 4. prepare_data Module Tests  (if available)
# ===========================================================================

class TestPrepareDataModule:
    """Tests that the prepare_data module functions behave correctly.

    These are skipped if prepare_data.py has not been created yet.
    """

    def test_load_qa_data_returns_list(self, prepare_data_module):
        """load_qa_data() must return a list of dicts."""
        data = prepare_data_module.load_qa_data()
        assert isinstance(data, list)
        assert len(data) == EXPECTED_QA_COUNT
        assert isinstance(data[0], dict)

    def test_load_cot_data_returns_list(self, prepare_data_module):
        """load_cot_data() must return a list of dicts."""
        data = prepare_data_module.load_cot_data()
        assert isinstance(data, list)
        assert len(data) == EXPECTED_COT_COUNT
        assert isinstance(data[0], dict)

    def test_format_qa_example_structure(self, prepare_data_module):
        """format_qa_example() must return a dict with 3 messages."""
        qa = prepare_data_module.load_qa_data()
        formatted = prepare_data_module.format_qa_example(qa[0])
        assert "messages" in formatted
        assert len(formatted["messages"]) == 3
        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][1]["role"] == "user"
        assert formatted["messages"][2]["role"] == "assistant"

    def test_format_cot_example_has_think_tags(self, prepare_data_module):
        """format_cot_example() must wrap reasoning in <think> tags."""
        cot = prepare_data_module.load_cot_data()
        formatted = prepare_data_module.format_cot_example(cot[0])
        assistant_content = formatted["messages"][2]["content"]
        assert "<think>" in assistant_content
        assert "</think>" in assistant_content

    def test_create_dataset_returns_correct_length(self, prepare_data_module):
        """create_dataset() must return a DatasetDict with 148 train examples."""
        dataset = prepare_data_module.create_dataset()
        assert len(dataset["train"]) == EXPECTED_TOTAL_COUNT


# ===========================================================================
# 5. Integration Tests  (require GPU)
# ===========================================================================

@pytest.mark.gpu
class TestGPUIntegration:
    """End-to-end tests that require a CUDA GPU. Skip on login/CPU nodes.

    Run with:  pytest test_training.py -m gpu -v
    """

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        """Skip entire class if no CUDA GPU is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("No CUDA GPU available")
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_loads_with_unsloth(self):
        """The Qwen model must load via Unsloth's FastLanguageModel."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-0.6B",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        assert model is not None
        assert tokenizer is not None

    def test_lora_applied_has_trainable_params(self):
        """After applying LoRA, the model must have trainable parameters."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-0.6B",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable > 0, "No trainable parameters after LoRA"
        assert trainable < total, "LoRA should freeze most parameters"
        # Trainable params should be a small fraction of total
        ratio = trainable / total
        assert ratio < 0.1, f"Trainable ratio {ratio:.4f} seems too high for LoRA"

    def test_single_forward_pass(self):
        """A single forward pass through the model must complete without error."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-0.6B",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        # Tokenize a simple input
        inputs = tokenizer(
            "What is gold mineralisation?",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(model.device)

        # Forward pass (no gradient needed for this test)
        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # batch size 1

    def test_single_training_step(self):
        """A single SFT training step must complete without error."""
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-0.6B",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        # Minimal dataset: 2 examples
        examples = [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "What is pyrite?"},
                    {"role": "assistant", "content": "Pyrite is an iron sulphide mineral (FeS2)."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "What is quartz?"},
                    {"role": "assistant", "content": "Quartz is silicon dioxide (SiO2), one of the most common minerals."},
                ]
            },
        ]
        dataset = Dataset.from_list(examples)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                max_steps=1,
                per_device_train_batch_size=2,
                learning_rate=2e-4,
                logging_steps=1,
                bf16=False,
                fp16=False,
                max_seq_length=512,
                seed=42,
            )
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            result = trainer.train()

        assert result is not None
        assert result.training_loss is not None
        assert result.training_loss > 0, "Training loss should be positive"
