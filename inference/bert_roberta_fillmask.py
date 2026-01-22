# -*- coding: utf-8 -*-
"""BERT/RoBERTa Fill-Mask Inference Script

Hugging Face pipeline-based infilling using masked language models.
Supports BERT, RoBERTa, DistilBERT, ALBERT, and other MLM architectures.

Usage:
    python bert_roberta_fillmask.py --text "The capital of France is [MASK]."
    python bert_roberta_fillmask.py --model roberta-base --text "The <mask> sat on the mat."
    python bert_roberta_fillmask.py --interactive --input texts.csv
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Union

import torch
from transformers import (
    pipeline,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)


# =============================================================================
# Configuration Dataclass
# =============================================================================

class ModelFamily(str, Enum):
    """Supported model families."""
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    ALBERT = "albert"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    XLMROBERTA = "xlm-roberta"
    CAMEMBERT = "camembert"
    CUSTOM = "custom"


class OutputFormat(str, Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"


class DeviceType(str, Enum):
    """Device type options."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class PrecisionType(str, Enum):
    """Model precision options."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ModelConfig:
    """Model loading configuration."""
    # Model identification
    model_name_or_path: str = "bert-base-uncased"
    model_family: ModelFamily = ModelFamily.BERT
    revision: str = "main"

    # Tokenizer options
    tokenizer_name_or_path: Optional[str] = None
    use_fast_tokenizer: bool = True
    add_prefix_space: bool = False

    # Model loading options
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    cache_dir: Optional[str] = None
    local_files_only: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Device settings
    device: DeviceType = DeviceType.AUTO
    device_map: Optional[str] = None
    precision: PrecisionType = PrecisionType.FP32

    # Generation parameters
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0

    # Batching
    batch_size: int = 1

    # Targets (for specific token predictions)
    targets: Optional[List[str]] = None


@dataclass
class InputConfig:
    """Input configuration."""
    # Direct text input
    text: Optional[str] = None
    texts: Optional[List[str]] = None

    # File input
    input_file: Optional[str] = None
    input_format: str = "csv"
    text_column: str = "text"
    id_column: Optional[str] = "id"

    # Masking options
    mask_token: Optional[str] = None  # Auto-detect from model if None
    mask_random_words: bool = False
    n_masks: int = 1
    mask_pattern: Optional[str] = None  # Regex pattern for words to mask

    # Filtering
    max_length: int = 512
    truncate: bool = True
    skip_empty: bool = True

    # Sampling
    shuffle: bool = False
    sample_size: Optional[int] = None
    start_index: int = 0
    end_index: Optional[int] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    output_file: Optional[str] = None
    output_format: OutputFormat = OutputFormat.TEXT

    # Display options
    show_scores: bool = True
    show_token_ids: bool = False
    show_sequence: bool = True
    colorize: bool = True
    verbose: bool = False
    quiet: bool = False

    # Formatting
    score_precision: int = 4
    indent: int = 2


@dataclass
class FillMaskConfig:
    """Complete configuration for fill-mask inference."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Reproducibility
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FillMaskConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            inference=InferenceConfig(**d.get("inference", {})),
            input=InputConfig(**d.get("input", {})),
            output=OutputConfig(**d.get("output", {})),
            seed=d.get("seed"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "FillMaskConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def to_json_file(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.BLUE = cls.YELLOW = cls.RED = ""
        cls.BOLD = cls.DIM = cls.RESET = ""


# =============================================================================
# Fill-Mask Pipeline
# =============================================================================

class FillMaskPipeline:
    """Wrapper around Hugging Face fill-mask pipeline with extended functionality."""

    # Mask tokens by model family
    MASK_TOKENS = {
        ModelFamily.BERT: "[MASK]",
        ModelFamily.ROBERTA: "<mask>",
        ModelFamily.DISTILBERT: "[MASK]",
        ModelFamily.ALBERT: "[MASK]",
        ModelFamily.ELECTRA: "[MASK]",
        ModelFamily.DEBERTA: "[MASK]",
        ModelFamily.XLMROBERTA: "<mask>",
        ModelFamily.CAMEMBERT: "<mask>",
    }

    def __init__(self, config: FillMaskConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.device = None
        self.mask_token = None

        self._setup_seed()
        self._load_pipeline()

    def _setup_seed(self):
        """Set random seed for reproducibility."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

    def _resolve_device(self) -> Union[str, int]:
        """Resolve device from config."""
        device_type = self.config.inference.device

        if device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif device_type == DeviceType.CUDA:
            return 0 if torch.cuda.is_available() else "cpu"
        elif device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def _get_torch_dtype(self):
        """Get torch dtype from precision config."""
        precision = self.config.inference.precision
        if precision == PrecisionType.FP16:
            return torch.float16
        elif precision == PrecisionType.BF16:
            return torch.bfloat16
        else:
            return torch.float32

    def _load_pipeline(self):
        """Load the fill-mask pipeline."""
        model_cfg = self.config.model
        inf_cfg = self.config.inference

        if not self.config.output.quiet:
            print(f"Loading model: {model_cfg.model_name_or_path}")

        # Resolve device
        self.device = self._resolve_device()

        # Tokenizer path
        tokenizer_path = model_cfg.tokenizer_name_or_path or model_cfg.model_name_or_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=model_cfg.use_fast_tokenizer,
            add_prefix_space=model_cfg.add_prefix_space,
            trust_remote_code=model_cfg.trust_remote_code,
            revision=model_cfg.revision,
            cache_dir=model_cfg.cache_dir,
            local_files_only=model_cfg.local_files_only,
            token=model_cfg.use_auth_token,
        )

        # Determine mask token
        if self.config.input.mask_token:
            self.mask_token = self.config.input.mask_token
        elif self.tokenizer.mask_token:
            self.mask_token = self.tokenizer.mask_token
        else:
            self.mask_token = self.MASK_TOKENS.get(model_cfg.model_family, "[MASK]")

        # Quantization config
        quantization_config = None
        if model_cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
            )
        elif model_cfg.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Model kwargs
        model_kwargs = {
            "trust_remote_code": model_cfg.trust_remote_code,
            "revision": model_cfg.revision,
            "cache_dir": model_cfg.cache_dir,
            "local_files_only": model_cfg.local_files_only,
            "token": model_cfg.use_auth_token,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self._get_torch_dtype()

        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_cfg.model_name_or_path,
            **model_kwargs
        )

        # Create pipeline
        pipe_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "top_k": inf_cfg.top_k,
        }

        if not quantization_config:
            pipe_kwargs["device"] = self.device

        self.pipe = pipeline("fill-mask", **pipe_kwargs)

        if not self.config.output.quiet:
            device_name = self.device if isinstance(self.device, str) else f"cuda:{self.device}"
            print(f"Model loaded on: {device_name}")
            print(f"Mask token: {self.mask_token}")

    def _mask_random_words(self, text: str) -> str:
        """Randomly mask words in text."""
        input_cfg = self.config.input

        # Find word positions
        words = list(re.finditer(r'\b\w+\b', text))
        if not words:
            return text

        n_masks = min(input_cfg.n_masks, len(words))
        if n_masks == 0:
            return text

        # Select positions to mask
        if input_cfg.mask_pattern:
            # Filter words matching pattern
            pattern = re.compile(input_cfg.mask_pattern)
            matching = [w for w in words if pattern.match(w.group())]
            if matching:
                words = matching

        selected = sorted(random.sample(range(len(words)), n_masks))

        # Build masked text
        result = []
        prev_end = 0
        for idx in selected:
            word = words[idx]
            result.append(text[prev_end:word.start()])
            result.append(self.mask_token)
            prev_end = word.end()
        result.append(text[prev_end:])

        return "".join(result)

    def _prepare_text(self, text: str) -> str:
        """Prepare text for inference."""
        input_cfg = self.config.input

        # Truncate if needed
        if input_cfg.truncate and len(text) > input_cfg.max_length:
            text = text[:input_cfg.max_length]

        # Mask random words if configured
        if input_cfg.mask_random_words and self.mask_token not in text:
            text = self._mask_random_words(text)

        return text

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Run fill-mask prediction on a single text."""
        text = self._prepare_text(text)

        if self.mask_token not in text:
            return []

        inf_cfg = self.config.inference

        # Run pipeline
        results = self.pipe(
            text,
            top_k=inf_cfg.top_k,
            targets=inf_cfg.targets,
        )

        # Handle single mask vs multiple masks
        if isinstance(results[0], dict):
            results = [results]

        return results

    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Run fill-mask prediction on multiple texts."""
        prepared = [self._prepare_text(t) for t in texts]
        prepared = [t for t in prepared if self.mask_token in t]

        if not prepared:
            return []

        inf_cfg = self.config.inference

        all_results = []
        for text in prepared:
            results = self.pipe(
                text,
                top_k=inf_cfg.top_k,
                targets=inf_cfg.targets,
            )
            if isinstance(results[0], dict):
                results = [results]
            all_results.append(results)

        return all_results

    def format_result(self, text: str, results: List[Dict[str, Any]],
                      entry_id: Optional[str] = None) -> str:
        """Format prediction results for display."""
        out_cfg = self.config.output

        if out_cfg.output_format == OutputFormat.JSON:
            output = {
                "input": text,
                "predictions": results,
            }
            if entry_id:
                output["id"] = entry_id
            return json.dumps(output, indent=out_cfg.indent)

        elif out_cfg.output_format == OutputFormat.JSONL:
            output = {
                "input": text,
                "predictions": results,
            }
            if entry_id:
                output["id"] = entry_id
            return json.dumps(output)

        elif out_cfg.output_format == OutputFormat.CSV:
            rows = []
            for mask_idx, mask_results in enumerate(results):
                for pred in mask_results:
                    row = {
                        "input": text,
                        "mask_index": mask_idx,
                        "token": pred["token_str"],
                        "score": pred["score"],
                    }
                    if entry_id:
                        row["id"] = entry_id
                    if out_cfg.show_sequence:
                        row["sequence"] = pred.get("sequence", "")
                    rows.append(row)
            return rows  # Return list for CSV writer

        else:  # TEXT format
            lines = []

            if entry_id:
                lines.append(f"ID: {entry_id}")

            lines.append(f"Input: {text}")
            lines.append("")

            for mask_idx, mask_results in enumerate(results):
                if len(results) > 1:
                    lines.append(f"Mask {mask_idx + 1}:")

                for i, pred in enumerate(mask_results, 1):
                    token = pred["token_str"]
                    score = pred["score"]

                    if out_cfg.colorize:
                        token_display = f"{Colors.GREEN}{Colors.BOLD}{token}{Colors.RESET}"
                    else:
                        token_display = token

                    line = f"  {i}. {token_display}"

                    if out_cfg.show_scores:
                        score_str = f"{score:.{out_cfg.score_precision}f}"
                        if out_cfg.colorize:
                            line += f" {Colors.DIM}(score: {score_str}){Colors.RESET}"
                        else:
                            line += f" (score: {score_str})"

                    if out_cfg.show_token_ids:
                        line += f" [id: {pred.get('token', 'N/A')}]"

                    lines.append(line)

                    if out_cfg.show_sequence:
                        seq = pred.get("sequence", "")
                        if out_cfg.colorize:
                            lines.append(f"     {Colors.DIM}{seq}{Colors.RESET}")
                        else:
                            lines.append(f"     {seq}")

                if mask_idx < len(results) - 1:
                    lines.append("")

            return "\n".join(lines)


# =============================================================================
# CLI Argument Parser
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="BERT/RoBERTa Fill-Mask Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with BERT
  %(prog)s --text "The capital of France is [MASK]."

  # Using RoBERTa
  %(prog)s --model roberta-base --text "The <mask> sat on the mat."

  # Process CSV file
  %(prog)s --input texts.csv --text-column content --top-k 10

  # Random masking
  %(prog)s --text "The quick brown fox jumps over the lazy dog" --mask-random --n-masks 2

  # JSON output
  %(prog)s --text "Hello [MASK]" --output-format json

  # Load config from file
  %(prog)s --config my_config.json
        """
    )

    # ===================
    # Config file
    # ===================
    parser.add_argument(
        "--config", "-c",
        type=str,
        metavar="PATH",
        help="Load configuration from JSON file"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        metavar="PATH",
        help="Save current configuration to JSON file and exit"
    )

    # ===================
    # Model Configuration
    # ===================
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        "--model", "-m",
        type=str,
        default="bert-base-uncased",
        metavar="NAME",
        help="Model name or path (default: bert-base-uncased)"
    )
    model_group.add_argument(
        "--model-family",
        type=str,
        choices=[f.value for f in ModelFamily],
        default="bert",
        help="Model family for mask token detection (default: bert)"
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision/branch (default: main)"
    )
    model_group.add_argument(
        "--tokenizer",
        type=str,
        metavar="NAME",
        help="Tokenizer name (defaults to model name)"
    )
    model_group.add_argument(
        "--no-fast-tokenizer",
        action="store_true",
        help="Disable fast tokenizer"
    )
    model_group.add_argument(
        "--add-prefix-space",
        action="store_true",
        help="Add prefix space for tokenization (RoBERTa-style)"
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading custom model code from Hub"
    )
    model_group.add_argument(
        "--auth-token",
        type=str,
        metavar="TOKEN",
        help="Hugging Face authentication token"
    )
    model_group.add_argument(
        "--cache-dir",
        type=str,
        metavar="PATH",
        help="Cache directory for models"
    )
    model_group.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use local files, no downloads"
    )

    # ===================
    # Quantization
    # ===================
    quant_group = parser.add_argument_group("Quantization")

    quant_group.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    quant_group.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    quant_group.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit (default: float16)"
    )
    quant_group.add_argument(
        "--bnb-4bit-quant-type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type (default: nf4)"
    )
    quant_group.add_argument(
        "--bnb-4bit-use-double-quant",
        action="store_true",
        help="Use double quantization for 4-bit"
    )

    # ===================
    # Inference Configuration
    # ===================
    inf_group = parser.add_argument_group("Inference Configuration")

    inf_group.add_argument(
        "--device", "-d",
        type=str,
        choices=[d.value for d in DeviceType],
        default="auto",
        help="Device to use (default: auto)"
    )
    inf_group.add_argument(
        "--device-map",
        type=str,
        help="Device map for model parallelism"
    )
    inf_group.add_argument(
        "--precision",
        type=str,
        choices=[p.value for p in PrecisionType],
        default="fp32",
        help="Model precision (default: fp32)"
    )
    inf_group.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top predictions to return (default: 5)"
    )
    inf_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) filtering threshold (default: 1.0)"
    )
    inf_group.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    inf_group.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    inf_group.add_argument(
        "--targets",
        type=str,
        nargs="+",
        metavar="TOKEN",
        help="Specific target tokens to score"
    )

    # ===================
    # Input Configuration
    # ===================
    input_group = parser.add_argument_group("Input Configuration")

    input_group.add_argument(
        "--text",
        type=str,
        help="Direct text input with mask token"
    )
    input_group.add_argument(
        "--texts",
        type=str,
        nargs="+",
        help="Multiple text inputs"
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        metavar="PATH",
        help="Input file path (CSV, JSON, JSONL, or TXT)"
    )
    input_group.add_argument(
        "--input-format",
        type=str,
        default="csv",
        choices=["csv", "json", "jsonl", "txt"],
        help="Input file format (default: csv)"
    )
    input_group.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text in CSV/JSON (default: text)"
    )
    input_group.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name for ID in CSV/JSON (default: id)"
    )
    input_group.add_argument(
        "--mask-token",
        type=str,
        help="Override mask token (auto-detected if not set)"
    )
    input_group.add_argument(
        "--mask-random", "-r",
        action="store_true",
        help="Randomly mask words in input text"
    )
    input_group.add_argument(
        "--n-masks", "-n",
        type=int,
        default=1,
        help="Number of words to mask when using --mask-random (default: 1)"
    )
    input_group.add_argument(
        "--mask-pattern",
        type=str,
        metavar="REGEX",
        help="Only mask words matching this regex pattern"
    )
    input_group.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum input length in characters (default: 512)"
    )
    input_group.add_argument(
        "--no-truncate",
        action="store_true",
        help="Don't truncate long inputs"
    )
    input_group.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="Skip empty inputs (default: True)"
    )
    input_group.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle input texts"
    )
    input_group.add_argument(
        "--sample-size",
        type=int,
        metavar="N",
        help="Only process N random samples"
    )
    input_group.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this index (default: 0)"
    )
    input_group.add_argument(
        "--end-index",
        type=int,
        help="Stop processing at this index"
    )

    # ===================
    # Output Configuration
    # ===================
    output_group = parser.add_argument_group("Output Configuration")

    output_group.add_argument(
        "--output", "-o",
        type=str,
        metavar="PATH",
        help="Output file path"
    )
    output_group.add_argument(
        "--output-format", "-f",
        type=str,
        choices=[f.value for f in OutputFormat],
        default="text",
        help="Output format (default: text)"
    )
    output_group.add_argument(
        "--no-scores",
        action="store_true",
        help="Don't show prediction scores"
    )
    output_group.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Show token IDs in output"
    )
    output_group.add_argument(
        "--no-sequence",
        action="store_true",
        help="Don't show full sequences in output"
    )
    output_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    output_group.add_argument(
        "--score-precision",
        type=int,
        default=4,
        help="Decimal places for scores (default: 4)"
    )
    output_group.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level (default: 2)"
    )

    # ===================
    # Reproducibility
    # ===================
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducibility"
    )

    return parser


def args_to_config(args: argparse.Namespace) -> FillMaskConfig:
    """Convert parsed arguments to FillMaskConfig."""

    # Load base config from file if provided
    if args.config:
        config = FillMaskConfig.from_json_file(args.config)
    else:
        config = FillMaskConfig()

    # Override with CLI arguments
    # Model config
    config.model.model_name_or_path = args.model
    config.model.model_family = ModelFamily(args.model_family)
    config.model.revision = args.revision
    config.model.tokenizer_name_or_path = args.tokenizer
    config.model.use_fast_tokenizer = not args.no_fast_tokenizer
    config.model.add_prefix_space = args.add_prefix_space
    config.model.trust_remote_code = args.trust_remote_code
    config.model.use_auth_token = args.auth_token
    config.model.cache_dir = args.cache_dir
    config.model.local_files_only = args.local_files_only
    config.model.load_in_8bit = args.load_in_8bit
    config.model.load_in_4bit = args.load_in_4bit
    config.model.bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype
    config.model.bnb_4bit_quant_type = args.bnb_4bit_quant_type
    config.model.bnb_4bit_use_double_quant = args.bnb_4bit_use_double_quant

    # Inference config
    config.inference.device = DeviceType(args.device)
    config.inference.device_map = args.device_map
    config.inference.precision = PrecisionType(args.precision)
    config.inference.top_k = args.top_k
    config.inference.top_p = args.top_p
    config.inference.temperature = args.temperature
    config.inference.batch_size = args.batch_size
    config.inference.targets = args.targets

    # Input config
    config.input.text = args.text
    config.input.texts = args.texts
    config.input.input_file = args.input
    config.input.input_format = args.input_format
    config.input.text_column = args.text_column
    config.input.id_column = args.id_column
    config.input.mask_token = args.mask_token
    config.input.mask_random_words = args.mask_random
    config.input.n_masks = args.n_masks
    config.input.mask_pattern = args.mask_pattern
    config.input.max_length = args.max_length
    config.input.truncate = not args.no_truncate
    config.input.skip_empty = args.skip_empty
    config.input.shuffle = args.shuffle
    config.input.sample_size = args.sample_size
    config.input.start_index = args.start_index
    config.input.end_index = args.end_index

    # Output config
    config.output.output_file = args.output
    config.output.output_format = OutputFormat(args.output_format)
    config.output.show_scores = not args.no_scores
    config.output.show_token_ids = args.show_token_ids
    config.output.show_sequence = not args.no_sequence
    config.output.colorize = not args.no_color
    config.output.verbose = args.verbose
    config.output.quiet = args.quiet
    config.output.score_precision = args.score_precision
    config.output.indent = args.indent

    # Reproducibility
    config.seed = args.seed

    return config


# =============================================================================
# Input Loading
# =============================================================================

def load_inputs(config: FillMaskConfig) -> List[Dict[str, Any]]:
    """Load inputs from various sources."""
    input_cfg = config.input
    entries = []

    # Direct text input
    if input_cfg.text:
        entries.append({"id": "cli_input", "text": input_cfg.text})

    # Multiple texts
    if input_cfg.texts:
        for i, text in enumerate(input_cfg.texts):
            entries.append({"id": f"cli_input_{i}", "text": text})

    # File input
    if input_cfg.input_file:
        if input_cfg.input_format == "csv":
            with open(input_cfg.input_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = {
                        "text": row.get(input_cfg.text_column, ""),
                    }
                    if input_cfg.id_column and input_cfg.id_column in row:
                        entry["id"] = row[input_cfg.id_column]
                    entries.append(entry)

        elif input_cfg.input_format == "json":
            with open(input_cfg.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            entries.append({"text": item})
                        else:
                            entries.append({
                                "text": item.get(input_cfg.text_column, ""),
                                "id": item.get(input_cfg.id_column),
                            })

        elif input_cfg.input_format == "jsonl":
            with open(input_cfg.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        entries.append({
                            "text": item.get(input_cfg.text_column, ""),
                            "id": item.get(input_cfg.id_column),
                        })

        elif input_cfg.input_format == "txt":
            with open(input_cfg.input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line or not input_cfg.skip_empty:
                        entries.append({"id": f"line_{i}", "text": line})

    # Filter empty
    if input_cfg.skip_empty:
        entries = [e for e in entries if e.get("text", "").strip()]

    # Shuffle
    if input_cfg.shuffle:
        random.shuffle(entries)

    # Slice
    start = input_cfg.start_index
    end = input_cfg.end_index if input_cfg.end_index else len(entries)
    entries = entries[start:end]

    # Sample
    if input_cfg.sample_size and input_cfg.sample_size < len(entries):
        entries = random.sample(entries, input_cfg.sample_size)

    return entries


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    # Convert args to config
    config = args_to_config(args)

    # Save config if requested
    if args.save_config:
        config.to_json_file(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
        return

    # Disable colors if requested
    if not config.output.colorize:
        Colors.disable()

    # Check input
    if not config.input.text and not config.input.texts and not config.input.input_file:
        parser.print_help()
        print("\nError: No input provided. Use --text, --texts, or --input")
        sys.exit(1)

    # Load pipeline
    pipe = FillMaskPipeline(config)

    # Load inputs
    entries = load_inputs(config)

    if not entries:
        print("No inputs to process")
        return

    if not config.output.quiet:
        print(f"\nProcessing {len(entries)} input(s)...")
        print("=" * 80)

    # Prepare output
    output_handle = None
    csv_writer = None

    if config.output.output_file:
        output_handle = open(config.output.output_file, "w", encoding="utf-8")
        if config.output.output_format == OutputFormat.CSV:
            fieldnames = ["id", "input", "mask_index", "token", "score"]
            if config.output.show_sequence:
                fieldnames.append("sequence")
            csv_writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            csv_writer.writeheader()

    # Process entries
    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        entry_id = entry.get("id")

        results = pipe.predict(text)

        if not results:
            if config.output.verbose:
                print(f"[{i+1}/{len(entries)}] No mask token found in: {text[:50]}...")
            continue

        formatted = pipe.format_result(text, results, entry_id)

        if config.output.output_format == OutputFormat.CSV:
            if csv_writer:
                for row in formatted:
                    csv_writer.writerow(row)
        else:
            if output_handle:
                output_handle.write(formatted + "\n\n")

            if not config.output.quiet:
                print(f"\n[{i+1}/{len(entries)}]")
                print(formatted)
                print("-" * 80)

    if output_handle:
        output_handle.close()
        if not config.output.quiet:
            print(f"\nOutput saved to: {config.output.output_file}")

    if not config.output.quiet:
        print("\nDone!")


if __name__ == "__main__":
    main()
