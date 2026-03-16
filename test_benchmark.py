"""
Tests for the multi-model benchmark pipeline.

Covers model_registry.py, benchmark.py helper functions, and evaluate.py
log_to_wandb changes. All tests run without GPU.

Run:
    pytest test_benchmark.py -v
"""

import inspect
import json
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_V2 = os.path.join(BASE_DIR, "training_splits_v2")


# ===========================================================================
# 1. Model Registry Tests
# ===========================================================================

class TestModelRegistry:
    """Tests for model_registry.py — model configs and helper functions."""

    def test_registry_not_empty(self):
        from model_registry import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) > 0, "Registry should have at least one model"

    def test_registry_has_five_dense_models(self):
        from model_registry import MODEL_REGISTRY
        assert len(MODEL_REGISTRY) == 5, (
            f"Expected 5 dense Qwen 3.5 models, got {len(MODEL_REGISTRY)}"
        )

    def test_all_models_have_required_fields(self):
        from model_registry import MODEL_REGISTRY
        required = {"model_id", "params_b", "vram_lora_gb", "batch_size", "is_vlm"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required - set(cfg.keys())
            assert not missing, (
                f"Model '{name}' missing fields: {missing}"
            )

    def test_model_ids_contain_qwen35(self):
        from model_registry import MODEL_REGISTRY
        for name, cfg in MODEL_REGISTRY.items():
            assert "Qwen3.5" in cfg["model_id"], (
                f"Model '{name}' ID should contain 'Qwen3.5', got '{cfg['model_id']}'"
            )

    def test_params_are_positive(self):
        from model_registry import MODEL_REGISTRY
        for name, cfg in MODEL_REGISTRY.items():
            assert cfg["params_b"] > 0, f"Model '{name}' params_b must be positive"

    def test_vram_estimates_are_positive(self):
        from model_registry import MODEL_REGISTRY
        for name, cfg in MODEL_REGISTRY.items():
            assert cfg["vram_lora_gb"] > 0, (
                f"Model '{name}' vram_lora_gb must be positive"
            )

    def test_batch_sizes_are_positive(self):
        from model_registry import MODEL_REGISTRY
        for name, cfg in MODEL_REGISTRY.items():
            assert cfg["batch_size"] > 0, (
                f"Model '{name}' batch_size must be positive"
            )

    def test_vram_increases_with_params(self):
        """Larger models should require more VRAM."""
        from model_registry import MODEL_REGISTRY
        items = sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["params_b"])
        for i in range(1, len(items)):
            prev_name, prev_cfg = items[i - 1]
            curr_name, curr_cfg = items[i]
            assert curr_cfg["vram_lora_gb"] >= prev_cfg["vram_lora_gb"], (
                f"{curr_name} ({curr_cfg['vram_lora_gb']}GB) should need >= "
                f"VRAM than {prev_name} ({prev_cfg['vram_lora_gb']}GB)"
            )

    def test_batch_sizes_decrease_with_model_size(self):
        """Larger models should have equal or smaller batch sizes."""
        from model_registry import MODEL_REGISTRY
        items = sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["params_b"])
        for i in range(1, len(items)):
            prev_cfg = items[i - 1][1]
            curr_name = items[i][0]
            curr_cfg = items[i][1]
            assert curr_cfg["batch_size"] <= prev_cfg["batch_size"], (
                f"{curr_name} batch_size should be <= {prev_cfg['batch_size']}"
            )

    def test_effective_batch_size_constant(self):
        """All models should have the same effective batch size."""
        from model_registry import get_model_config, get_all_model_names
        from model_registry import EFFECTIVE_BATCH_SIZE

        for name in get_all_model_names():
            cfg = get_model_config(name)
            effective = cfg["batch_size"] * cfg["gradient_accumulation_steps"]
            assert effective == EFFECTIVE_BATCH_SIZE, (
                f"{name}: effective batch size {effective} != "
                f"expected {EFFECTIVE_BATCH_SIZE}"
            )

    def test_lora_target_modules_not_empty(self):
        from model_registry import LORA_TARGET_MODULES
        assert len(LORA_TARGET_MODULES) > 0

    def test_lora_target_modules_has_attention_and_mlp(self):
        from model_registry import LORA_TARGET_MODULES
        assert "q_proj" in LORA_TARGET_MODULES
        assert "v_proj" in LORA_TARGET_MODULES
        assert "gate_proj" in LORA_TARGET_MODULES

    def test_model_names_ordered_by_size(self):
        from model_registry import get_all_model_names, MODEL_REGISTRY
        names = get_all_model_names()
        params = [MODEL_REGISTRY[n]["params_b"] for n in names]
        assert params == sorted(params), "Models should be ordered by size"


class TestModelRegistryHelpers:
    """Tests for model_registry.py helper functions."""

    def test_get_model_config_valid(self):
        from model_registry import get_model_config
        cfg = get_model_config("Qwen3.5-0.8B")
        assert cfg["name"] == "Qwen3.5-0.8B"
        assert cfg["params_b"] == 0.8
        assert "gradient_accumulation_steps" in cfg

    def test_get_model_config_invalid_raises(self):
        from model_registry import get_model_config
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("NonexistentModel")

    def test_get_all_model_names_returns_list(self):
        from model_registry import get_all_model_names
        names = get_all_model_names()
        assert isinstance(names, list)
        assert len(names) == 5

    def test_get_models_fitting_vram_a100(self):
        from model_registry import get_models_fitting_vram
        models = get_models_fitting_vram(vram_gb=80)
        assert len(models) >= 4, "At least 4 models should fit on A100-80GB"
        assert "Qwen3.5-0.8B" in models
        assert "Qwen3.5-9B" in models

    def test_get_models_fitting_vram_small_gpu(self):
        from model_registry import get_models_fitting_vram
        models = get_models_fitting_vram(vram_gb=8)
        assert "Qwen3.5-0.8B" in models
        assert "Qwen3.5-2B" in models
        assert "Qwen3.5-27B" not in models

    def test_get_models_fitting_vram_tiny_gpu(self):
        from model_registry import get_models_fitting_vram
        models = get_models_fitting_vram(vram_gb=2)
        assert len(models) == 0

    def test_validate_model_names_valid(self):
        from model_registry import validate_model_names
        result = validate_model_names(["Qwen3.5-0.8B", "Qwen3.5-2B"])
        assert result == ["Qwen3.5-0.8B", "Qwen3.5-2B"]

    def test_validate_model_names_invalid_raises(self):
        from model_registry import validate_model_names
        with pytest.raises(ValueError, match="Unknown model"):
            validate_model_names(["Qwen3.5-0.8B", "FakeModel"])

    def test_resolve_model_path_local_exists(self):
        """Should return local path when model directory exists."""
        from model_registry import resolve_model_path
        # Qwen3.5-0.8B is pre-downloaded locally
        path = resolve_model_path("Qwen3.5-0.8B", local_model_dir="./models")
        expected_local = os.path.join("./models", "Qwen3.5-0.8B")
        if os.path.isdir(expected_local):
            assert path == expected_local
        else:
            # If not downloaded, should return model_id
            assert "Qwen3.5" in path

    def test_resolve_model_path_fallback_to_id(self):
        """Should return Unsloth model ID when no local copy exists."""
        from model_registry import resolve_model_path
        with tempfile.TemporaryDirectory() as tmp:
            path = resolve_model_path("Qwen3.5-0.8B", local_model_dir=tmp)
            assert path == "unsloth/Qwen3.5-0.8B"


# ===========================================================================
# 2. Benchmark Helper Tests
# ===========================================================================

class TestBenchmarkHelpers:
    """Tests for benchmark.py helper functions (no GPU needed)."""

    def test_flatten_eval_results(self):
        from benchmark import flatten_eval_results

        raw_summary = {
            "qa/rouge1_f1": 0.12345,
            "qa/rouge2_f1": 0.06789,
            "qa/rougeL_f1": 0.11111,
            "qa/bertscore_f1": 0.84567,
            "cot/rouge1_f1": 0.15,
            "cot/rouge2_f1": 0.08,
            "cot/rougeL_f1": 0.13,
            "cot/bertscore_f1": 0.82,
            "cot/think_tag_rate": 0.333,
            "hallucination/pass_rate": 0.6,
            "overall/weighted_score": 0.42,
        }
        flat = flatten_eval_results(raw_summary)

        assert flat["qa_rouge1"] == 0.1235  # rounded to 4 places
        assert flat["qa_rougeL"] == 0.1111
        assert flat["qa_bertscore_f1"] == 0.8457
        assert flat["cot_think_tag_rate"] == 0.333
        assert flat["hallucination_pass_rate"] == 0.6
        assert flat["overall_weighted_score"] == 0.42

    def test_flatten_eval_results_missing_keys(self):
        """Missing keys should default to 0."""
        from benchmark import flatten_eval_results
        flat = flatten_eval_results({})
        assert flat["qa_rouge1"] == 0
        assert flat["overall_weighted_score"] == 0

    def test_save_and_load_model_result(self):
        from benchmark import save_model_result

        result = {
            "model_name": "Qwen3.5-0.8B",
            "base_eval": {"qa_rougeL": 0.15, "overall_weighted_score": 0.3},
            "finetuned_eval": {"qa_rougeL": 0.25, "overall_weighted_score": 0.5},
            "training": {"epochs": 5, "final_train_loss": 1.5},
        }

        with tempfile.TemporaryDirectory() as tmp:
            save_model_result(tmp, "Qwen3.5-0.8B", result)
            path = os.path.join(tmp, "Qwen3.5-0.8B.json")
            assert os.path.isfile(path)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["model_name"] == "Qwen3.5-0.8B"
            assert loaded["base_eval"]["qa_rougeL"] == 0.15
            assert loaded["training"]["epochs"] == 5

    def test_save_benchmark_results(self):
        from benchmark import save_benchmark_results

        data = {
            "benchmark_id": "benchmark-test",
            "timestamp": "2026-03-15T12:00:00",
            "config": {"epochs": 5},
            "models": {
                "Qwen3.5-0.8B": {"base_eval": {"overall_weighted_score": 0.3}},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            save_benchmark_results(tmp, data)
            path = os.path.join(tmp, "benchmark_summary.json")
            assert os.path.isfile(path)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["benchmark_id"] == "benchmark-test"
            assert "Qwen3.5-0.8B" in loaded["models"]

    def test_cleanup_gpu_runs_without_error(self):
        """cleanup_gpu should not raise even without CUDA."""
        from benchmark import cleanup_gpu
        cleanup_gpu()  # Should not raise


class TestBenchmarkCLI:
    """Tests for benchmark.py CLI argument parsing."""

    def test_parse_args_defaults(self):
        from benchmark import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["benchmark.py"]
        try:
            args = parse_args()
            assert args.epochs == 5
            assert args.learning_rate == 2e-4
            assert args.lora_r == 16
            assert args.lora_alpha == 16
            assert args.max_seq_length == 2048
            assert args.models == [
                "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B",
                "Qwen3.5-9B", "Qwen3.5-27B",
            ]
            assert args.data_dir == "./training_splits_v2"
            assert args.results_dir == "./benchmark_results"
            assert args.seed == 3407
            assert args.skip_base_eval is False
        finally:
            sys.argv = old_argv

    def test_parse_args_custom_models(self):
        from benchmark import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = [
            "benchmark.py",
            "--models", "Qwen3.5-0.8B", "Qwen3.5-2B",
            "--epochs", "10",
            "--learning_rate", "1e-4",
        ]
        try:
            args = parse_args()
            assert args.models == ["Qwen3.5-0.8B", "Qwen3.5-2B"]
            assert args.epochs == 10
            assert args.learning_rate == 1e-4
        finally:
            sys.argv = old_argv

    def test_parse_args_skip_base_eval(self):
        from benchmark import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["benchmark.py", "--skip_base_eval"]
        try:
            args = parse_args()
            assert args.skip_base_eval is True
        finally:
            sys.argv = old_argv


# ===========================================================================
# 3. Results Format Tests
# ===========================================================================

class TestResultsFormat:
    """Tests that the benchmark results JSON has the expected structure."""

    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data matching the expected format."""
        return {
            "benchmark_id": "benchmark-20260315-120000",
            "timestamp": "2026-03-15T12:00:00",
            "config": {
                "epochs": 5,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 16,
                "max_seq_length": 2048,
                "effective_batch_size": 16,
                "data_dir": "./training_splits_v2",
                "seed": 3407,
            },
            "models": {
                "Qwen3.5-0.8B": {
                    "model_name": "Qwen3.5-0.8B",
                    "model_id": "unsloth/Qwen3.5-0.8B",
                    "params_b": 0.8,
                    "base_eval": {
                        "qa_rouge1": 0.12,
                        "qa_rouge2": 0.05,
                        "qa_rougeL": 0.10,
                        "qa_bertscore_f1": 0.80,
                        "cot_rouge1": 0.11,
                        "cot_rouge2": 0.04,
                        "cot_rougeL": 0.09,
                        "cot_bertscore_f1": 0.78,
                        "cot_think_tag_rate": 0.0,
                        "hallucination_pass_rate": 0.2,
                        "overall_weighted_score": 0.30,
                    },
                    "finetuned_eval": {
                        "qa_rouge1": 0.25,
                        "qa_rouge2": 0.12,
                        "qa_rougeL": 0.20,
                        "qa_bertscore_f1": 0.85,
                        "cot_rouge1": 0.22,
                        "cot_rouge2": 0.10,
                        "cot_rougeL": 0.18,
                        "cot_bertscore_f1": 0.83,
                        "cot_think_tag_rate": 0.5,
                        "hallucination_pass_rate": 0.6,
                        "overall_weighted_score": 0.50,
                    },
                    "training": {
                        "epochs": 5,
                        "final_train_loss": 1.5,
                        "wall_time_seconds": 300.0,
                        "trainable_params": 5000000,
                        "total_params": 800000000,
                        "trainable_pct": 0.63,
                        "batch_size": 4,
                        "effective_batch_size": 16,
                        "learning_rate": 2e-4,
                        "lora_r": 16,
                        "lora_alpha": 16,
                    },
                    "error": None,
                },
            },
        }

    def test_benchmark_data_has_required_top_keys(self, sample_benchmark_data):
        required = {"benchmark_id", "timestamp", "config", "models"}
        assert required.issubset(sample_benchmark_data.keys())

    def test_config_has_training_params(self, sample_benchmark_data):
        cfg = sample_benchmark_data["config"]
        assert "epochs" in cfg
        assert "learning_rate" in cfg
        assert "lora_r" in cfg
        assert "seed" in cfg

    def test_model_result_has_all_sections(self, sample_benchmark_data):
        model = sample_benchmark_data["models"]["Qwen3.5-0.8B"]
        assert "base_eval" in model
        assert "finetuned_eval" in model
        assert "training" in model
        assert "error" in model

    def test_eval_results_have_all_metrics(self, sample_benchmark_data):
        expected_metrics = {
            "qa_rouge1", "qa_rouge2", "qa_rougeL", "qa_bertscore_f1",
            "cot_rouge1", "cot_rouge2", "cot_rougeL", "cot_bertscore_f1",
            "cot_think_tag_rate", "hallucination_pass_rate",
            "overall_weighted_score",
        }
        for eval_type in ["base_eval", "finetuned_eval"]:
            metrics = sample_benchmark_data["models"]["Qwen3.5-0.8B"][eval_type]
            missing = expected_metrics - set(metrics.keys())
            assert not missing, f"{eval_type} missing metrics: {missing}"

    def test_training_results_have_expected_keys(self, sample_benchmark_data):
        training = sample_benchmark_data["models"]["Qwen3.5-0.8B"]["training"]
        expected = {
            "epochs", "final_train_loss", "wall_time_seconds",
            "trainable_params", "total_params", "trainable_pct",
            "batch_size", "effective_batch_size",
        }
        missing = expected - set(training.keys())
        assert not missing, f"Training results missing keys: {missing}"

    def test_results_json_round_trip(self, sample_benchmark_data):
        """Results should survive JSON serialization."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(sample_benchmark_data, f)
            path = f.name
        try:
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == sample_benchmark_data
        finally:
            os.unlink(path)

    def test_finetuned_better_than_base(self, sample_benchmark_data):
        """Fine-tuned model should score higher than base (in sample data)."""
        model = sample_benchmark_data["models"]["Qwen3.5-0.8B"]
        base_score = model["base_eval"]["overall_weighted_score"]
        ft_score = model["finetuned_eval"]["overall_weighted_score"]
        assert ft_score > base_score


# ===========================================================================
# 4. Comparison Table Tests
# ===========================================================================

class TestComparisonTable:
    """Tests for the terminal comparison table formatting."""

    def test_print_comparison_table_runs(self, capsys):
        from benchmark import print_comparison_table

        data = {
            "models": {
                "Qwen3.5-0.8B": {
                    "base_eval": {
                        "qa_rougeL": 0.10, "qa_bertscore_f1": 0.80,
                        "cot_rougeL": 0.09, "cot_bertscore_f1": 0.78,
                        "cot_think_tag_rate": 0.0,
                        "hallucination_pass_rate": 0.2,
                        "overall_weighted_score": 0.30,
                    },
                    "finetuned_eval": {
                        "qa_rougeL": 0.20, "qa_bertscore_f1": 0.85,
                        "cot_rougeL": 0.18, "cot_bertscore_f1": 0.83,
                        "cot_think_tag_rate": 0.5,
                        "hallucination_pass_rate": 0.6,
                        "overall_weighted_score": 0.50,
                    },
                    "training": {
                        "final_train_loss": 1.5,
                        "wall_time_seconds": 300,
                        "trainable_params": 5000000,
                        "trainable_pct": 0.63,
                    },
                },
            },
        }

        print_comparison_table(data)
        output = capsys.readouterr().out

        assert "BENCHMARK COMPARISON" in output
        assert "Qwen3.5-0.8B" in output
        assert "base" in output
        assert "finetuned" in output

    def test_print_comparison_table_handles_none(self, capsys):
        """Should handle models with None eval results (failed)."""
        from benchmark import print_comparison_table

        data = {
            "models": {
                "Qwen3.5-0.8B": {
                    "base_eval": None,
                    "finetuned_eval": None,
                    "training": None,
                },
            },
        }

        print_comparison_table(data)
        output = capsys.readouterr().out
        assert "BENCHMARK COMPARISON" in output


# ===========================================================================
# 5. Evaluate.py WandB Flag Tests
# ===========================================================================

class TestEvaluateWandbFlag:
    """Tests for the log_to_wandb parameter in evaluate.py."""

    def test_run_evaluation_has_log_to_wandb_param(self):
        """run_evaluation should accept log_to_wandb parameter."""
        from evaluate import run_evaluation
        sig = inspect.signature(run_evaluation)
        assert "log_to_wandb" in sig.parameters, (
            "run_evaluation must have 'log_to_wandb' parameter"
        )

    def test_log_to_wandb_defaults_to_true(self):
        """log_to_wandb should default to True for backwards compatibility."""
        from evaluate import run_evaluation
        sig = inspect.signature(run_evaluation)
        default = sig.parameters["log_to_wandb"].default
        assert default is True


# ===========================================================================
# 6. Data File Tests (v2 test data)
# ===========================================================================

class TestBenchmarkData:
    """Tests that the v2 data files needed for benchmarking exist."""

    def test_v2_qa_test_exists(self):
        path = os.path.join(DATA_DIR_V2, "qa_test.jsonl")
        assert os.path.isfile(path), f"Missing {path}"

    def test_v2_cot_test_exists(self):
        path = os.path.join(DATA_DIR_V2, "cot_test.jsonl")
        assert os.path.isfile(path), f"Missing {path}"

    def test_v2_hallucination_traps_exists(self):
        path = os.path.join(DATA_DIR_V2, "hallucination_traps.jsonl")
        assert os.path.isfile(path), f"Missing {path}"

    def test_v2_qa_test_records_valid(self):
        path = os.path.join(DATA_DIR_V2, "qa_test.jsonl")
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) > 0, "qa_test.jsonl should not be empty"
        for i, rec in enumerate(records):
            assert "question" in rec, f"Record {i} missing 'question'"
            assert "answer" in rec, f"Record {i} missing 'answer'"

    def test_v2_cot_test_records_valid(self):
        path = os.path.join(DATA_DIR_V2, "cot_test.jsonl")
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) > 0, "cot_test.jsonl should not be empty"
        for i, rec in enumerate(records):
            assert "question" in rec, f"Record {i} missing 'question'"
            assert "answer" in rec, f"Record {i} missing 'answer'"

    def test_v2_traps_have_required_fields(self):
        path = os.path.join(DATA_DIR_V2, "hallucination_traps.jsonl")
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) > 0
        for i, rec in enumerate(records):
            assert "question" in rec, f"Trap {i} missing 'question'"
            assert "category" in rec, f"Trap {i} missing 'category'"
            assert "acceptable_responses" in rec, (
                f"Trap {i} missing 'acceptable_responses'"
            )

    def test_processed_data_exists(self):
        """processed_data/ should exist (created by prepare_data.py)."""
        path = os.path.join(BASE_DIR, "processed_data")
        assert os.path.isdir(path), (
            f"Missing {path}. Run: python prepare_data.py"
        )


# ===========================================================================
# 7. Integration Tests (structure only, no GPU)
# ===========================================================================

class TestBenchmarkIntegration:
    """Structural integration tests — verify imports and function signatures."""

    def test_benchmark_imports_model_registry(self):
        """benchmark.py should be importable."""
        import benchmark
        assert hasattr(benchmark, "run_benchmark")
        assert hasattr(benchmark, "evaluate_model")
        assert hasattr(benchmark, "train_model")

    def test_benchmark_load_model_signature(self):
        from benchmark import load_model
        sig = inspect.signature(load_model)
        assert "model_path" in sig.parameters
        assert "max_seq_length" in sig.parameters

    def test_benchmark_apply_lora_signature(self):
        from benchmark import apply_lora
        sig = inspect.signature(apply_lora)
        assert "model" in sig.parameters
        assert "lora_r" in sig.parameters
        assert "lora_alpha" in sig.parameters

    def test_benchmark_evaluate_model_signature(self):
        from benchmark import evaluate_model
        sig = inspect.signature(evaluate_model)
        assert "model" in sig.parameters
        assert "tokenizer" in sig.parameters
        assert "data_dir" in sig.parameters

    def test_benchmark_train_model_signature(self):
        from benchmark import train_model
        sig = inspect.signature(train_model)
        params = set(sig.parameters.keys())
        expected = {
            "model", "tokenizer", "dataset", "model_name",
            "epochs", "learning_rate", "batch_size",
        }
        missing = expected - params
        assert not missing, f"train_model missing params: {missing}"

    def test_ensure_processed_data_skips_existing(self):
        """Should not rebuild if processed_data/ already exists."""
        from benchmark import ensure_processed_data
        processed = os.path.join(BASE_DIR, "processed_data")
        if os.path.isdir(processed):
            # Should return without error (no rebuild)
            ensure_processed_data(DATA_DIR_V2, processed)

    def test_model_registry_importable_from_benchmark(self):
        """benchmark.py should be able to import model_registry."""
        from model_registry import (
            get_model_config,
            get_all_model_names,
            get_models_fitting_vram,
            validate_model_names,
            resolve_model_path,
        )
        assert callable(get_model_config)
        assert callable(get_all_model_names)
