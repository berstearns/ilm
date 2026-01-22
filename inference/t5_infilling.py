# -*- coding: utf-8 -*-
"""T5/BART Text Infilling Inference Script

Hugging Face pipeline-based infilling using encoder-decoder models.
Supports T5, BART, mT5, FLAN-T5, and other seq2seq architectures.

T5 uses sentinel tokens (<extra_id_0>, <extra_id_1>, ...) for span corruption/infilling.
BART uses <mask> tokens for infilling.

Usage:
    python t5_infilling.py --text "The <extra_id_0> sat on the <extra_id_1>."
    python t5_infilling.py --model google/flan-t5-base --text "Fill: The <extra_id_0> is blue."
    python t5_infilling.py --model facebook/bart-base --text "The <mask> sat on the mat."
    python t5_infilling.py --interactive --input texts.csv
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
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig,
)


# =============================================================================
# Configuration Dataclass
# =============================================================================

class ModelFamily(str, Enum):
    """Supported model families."""
    T5 = "t5"
    T5_V1_1 = "t5-v1_1"
    FLAN_T5 = "flan-t5"
    MT5 = "mt5"
    BART = "bart"
    MBART = "mbart"
    PEGASUS = "pegasus"
    LED = "led"
    LONGT5 = "longt5"
    UL2 = "ul2"
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


class DecodingStrategy(str, Enum):
    """Decoding strategy options."""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    SAMPLING = "sampling"
    TOP_K = "top_k"
    TOP_P = "top_p"
    CONTRASTIVE = "contrastive"


class InfillingMode(str, Enum):
    """Infilling mode options."""
    SPAN_CORRUPTION = "span_corruption"  # T5-style with sentinel tokens
    MASK_INFILL = "mask_infill"  # BART-style with <mask>
    PREFIX_LM = "prefix_lm"  # UL2-style
    FILL_PROMPT = "fill_prompt"  # Instruction-based


@dataclass
class ModelConfig:
    """Model loading configuration."""
    # Model identification
    model_name_or_path: str = "t5-base"
    model_family: ModelFamily = ModelFamily.T5
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
class GenerationParams:
    """Generation/decoding configuration."""
    # Basic generation
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    max_length: Optional[int] = None

    # Decoding strategy
    decoding_strategy: DecodingStrategy = DecodingStrategy.GREEDY
    num_beams: int = 4
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    early_stopping: bool = True

    # Sampling parameters
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_repetition_penalty: float = 1.0
    length_penalty: float = 1.0

    # Multiple outputs
    num_return_sequences: int = 1

    # Contrastive search
    penalty_alpha: float = 0.0

    # Constraints
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Device settings
    device: DeviceType = DeviceType.AUTO
    device_map: Optional[str] = None
    precision: PrecisionType = PrecisionType.FP32

    # Generation
    generation: GenerationParams = field(default_factory=GenerationParams)

    # Batching
    batch_size: int = 1

    # Infilling mode
    infilling_mode: InfillingMode = InfillingMode.SPAN_CORRUPTION


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

    # Sentinel/mask tokens
    sentinel_token_pattern: str = "<extra_id_{i}>"
    mask_token: str = "<mask>"
    num_sentinel_tokens: int = 100

    # Auto-masking options
    mask_random_spans: bool = False
    n_spans: int = 1
    mean_span_length: float = 3.0
    span_length_distribution: str = "geometric"  # geometric, uniform, fixed
    mask_pattern: Optional[str] = None  # Regex pattern for spans to mask

    # Prompting (for instruction models)
    prefix: str = ""
    suffix: str = ""
    prompt_template: Optional[str] = None  # e.g., "Fill in the blanks: {text}"

    # Input processing
    max_source_length: int = 512
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
    show_input: bool = True
    show_raw_output: bool = False
    show_reconstructed: bool = True
    show_scores: bool = False
    colorize: bool = True
    verbose: bool = False
    quiet: bool = False

    # Formatting
    score_precision: int = 4
    indent: int = 2


@dataclass
class T5InfillingConfig:
    """Complete configuration for T5 infilling inference."""
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
    def from_dict(cls, d: Dict[str, Any]) -> "T5InfillingConfig":
        """Create config from dictionary."""
        gen_params = GenerationParams(**d.get("inference", {}).get("generation", {}))
        inf_config = d.get("inference", {})
        inf_config["generation"] = gen_params

        return cls(
            model=ModelConfig(**d.get("model", {})),
            inference=InferenceConfig(**inf_config),
            input=InputConfig(**d.get("input", {})),
            output=OutputConfig(**d.get("output", {})),
            seed=d.get("seed"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "T5InfillingConfig":
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
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        for attr in ['GREEN', 'BLUE', 'YELLOW', 'RED', 'CYAN', 'MAGENTA',
                     'BOLD', 'DIM', 'UNDERLINE', 'RESET']:
            setattr(cls, attr, "")


# =============================================================================
# T5 Infilling Pipeline
# =============================================================================

class T5InfillingPipeline:
    """Wrapper around T5/seq2seq models for infilling tasks."""

    # Model family to default sentinel pattern
    SENTINEL_PATTERNS = {
        ModelFamily.T5: "<extra_id_{i}>",
        ModelFamily.T5_V1_1: "<extra_id_{i}>",
        ModelFamily.FLAN_T5: "<extra_id_{i}>",
        ModelFamily.MT5: "<extra_id_{i}>",
        ModelFamily.LONGT5: "<extra_id_{i}>",
        ModelFamily.UL2: "<extra_id_{i}>",
        ModelFamily.BART: "<mask>",
        ModelFamily.MBART: "<mask>",
        ModelFamily.PEGASUS: "<mask_1>",
    }

    def __init__(self, config: T5InfillingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.sentinel_pattern = None
        self.sentinel_tokens = []

        self._setup_seed()
        self._load_model()
        self._setup_sentinels()

    def _setup_seed(self):
        """Set random seed for reproducibility."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

    def _resolve_device(self) -> torch.device:
        """Resolve device from config."""
        device_type = self.config.inference.device

        if device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif device_type == DeviceType.CUDA:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_type == DeviceType.MPS:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _get_torch_dtype(self):
        """Get torch dtype from precision config."""
        precision = self.config.inference.precision
        if precision == PrecisionType.FP16:
            return torch.float16
        elif precision == PrecisionType.BF16:
            return torch.bfloat16
        else:
            return torch.float32

    def _load_model(self):
        """Load the T5/seq2seq model."""
        model_cfg = self.config.model

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
            trust_remote_code=model_cfg.trust_remote_code,
            revision=model_cfg.revision,
            cache_dir=model_cfg.cache_dir,
            local_files_only=model_cfg.local_files_only,
            token=model_cfg.use_auth_token,
        )

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_cfg.model_name_or_path,
            **model_kwargs
        )

        if not quantization_config:
            self.model.to(self.device)

        self.model.eval()

        if not self.config.output.quiet:
            print(f"Model loaded on: {self.device}")

    def _setup_sentinels(self):
        """Setup sentinel tokens based on model family."""
        input_cfg = self.config.input
        model_cfg = self.config.model

        # Determine sentinel pattern
        self.sentinel_pattern = input_cfg.sentinel_token_pattern
        if not self.sentinel_pattern:
            self.sentinel_pattern = self.SENTINEL_PATTERNS.get(
                model_cfg.model_family,
                "<extra_id_{i}>"
            )

        # Generate sentinel tokens list
        self.sentinel_tokens = []
        for i in range(input_cfg.num_sentinel_tokens):
            token = self.sentinel_pattern.format(i=i)
            self.sentinel_tokens.append(token)

        # Check which sentinels are in vocabulary
        valid_sentinels = []
        for token in self.sentinel_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                valid_sentinels.append(token)

        if valid_sentinels:
            self.sentinel_tokens = valid_sentinels
            if not self.config.output.quiet:
                print(f"Found {len(self.sentinel_tokens)} sentinel tokens")
        else:
            # Fall back to mask token
            if self.tokenizer.mask_token:
                self.sentinel_tokens = [self.tokenizer.mask_token]
            else:
                self.sentinel_tokens = [input_cfg.mask_token]
            if not self.config.output.quiet:
                print(f"Using mask token: {self.sentinel_tokens[0]}")

    def _get_generation_config(self) -> Dict[str, Any]:
        """Build generation kwargs from config."""
        gen_cfg = self.config.inference.generation

        kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "min_new_tokens": gen_cfg.min_new_tokens,
            "num_return_sequences": gen_cfg.num_return_sequences,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "length_penalty": gen_cfg.length_penalty,
        }

        if gen_cfg.max_length:
            kwargs["max_length"] = gen_cfg.max_length

        if gen_cfg.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = gen_cfg.no_repeat_ngram_size

        if gen_cfg.encoder_repetition_penalty != 1.0:
            kwargs["encoder_repetition_penalty"] = gen_cfg.encoder_repetition_penalty

        # Decoding strategy
        strategy = gen_cfg.decoding_strategy

        if strategy == DecodingStrategy.GREEDY:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1

        elif strategy == DecodingStrategy.BEAM_SEARCH:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = gen_cfg.num_beams
            kwargs["early_stopping"] = gen_cfg.early_stopping
            if gen_cfg.num_beam_groups > 1:
                kwargs["num_beam_groups"] = gen_cfg.num_beam_groups
                kwargs["diversity_penalty"] = gen_cfg.diversity_penalty

        elif strategy == DecodingStrategy.SAMPLING:
            kwargs["do_sample"] = True
            kwargs["temperature"] = gen_cfg.temperature

        elif strategy == DecodingStrategy.TOP_K:
            kwargs["do_sample"] = True
            kwargs["temperature"] = gen_cfg.temperature
            kwargs["top_k"] = gen_cfg.top_k

        elif strategy == DecodingStrategy.TOP_P:
            kwargs["do_sample"] = True
            kwargs["temperature"] = gen_cfg.temperature
            kwargs["top_p"] = gen_cfg.top_p
            kwargs["top_k"] = 0  # Disable top-k when using top-p

        elif strategy == DecodingStrategy.CONTRASTIVE:
            kwargs["do_sample"] = False
            kwargs["penalty_alpha"] = gen_cfg.penalty_alpha
            kwargs["top_k"] = gen_cfg.top_k

        # Additional sampling params
        if gen_cfg.do_sample or strategy in [DecodingStrategy.SAMPLING,
                                              DecodingStrategy.TOP_K,
                                              DecodingStrategy.TOP_P]:
            if gen_cfg.typical_p < 1.0:
                kwargs["typical_p"] = gen_cfg.typical_p
            if gen_cfg.epsilon_cutoff > 0:
                kwargs["epsilon_cutoff"] = gen_cfg.epsilon_cutoff
            if gen_cfg.eta_cutoff > 0:
                kwargs["eta_cutoff"] = gen_cfg.eta_cutoff

        # Constraints
        if gen_cfg.forced_bos_token_id is not None:
            kwargs["forced_bos_token_id"] = gen_cfg.forced_bos_token_id
        if gen_cfg.forced_eos_token_id is not None:
            kwargs["forced_eos_token_id"] = gen_cfg.forced_eos_token_id
        if gen_cfg.suppress_tokens:
            kwargs["suppress_tokens"] = gen_cfg.suppress_tokens
        if gen_cfg.begin_suppress_tokens:
            kwargs["begin_suppress_tokens"] = gen_cfg.begin_suppress_tokens

        return kwargs

    def _sample_span_lengths(self, num_spans: int, text_length: int) -> List[int]:
        """Sample span lengths based on distribution."""
        input_cfg = self.config.input
        mean_len = input_cfg.mean_span_length
        dist = input_cfg.span_length_distribution

        lengths = []
        for _ in range(num_spans):
            if dist == "fixed":
                length = int(mean_len)
            elif dist == "uniform":
                length = random.randint(1, int(2 * mean_len))
            else:  # geometric
                length = max(1, int(random.expovariate(1.0 / mean_len)))
            lengths.append(min(length, text_length // (num_spans + 1)))

        return lengths

    def _mask_random_spans(self, text: str) -> Tuple[str, List[str]]:
        """Randomly mask spans in text with sentinel tokens."""
        input_cfg = self.config.input
        words = text.split()

        if len(words) < 2:
            return text, []

        n_spans = min(input_cfg.n_spans, len(words) // 2, len(self.sentinel_tokens))
        if n_spans == 0:
            return text, []

        # Sample span lengths
        span_lengths = self._sample_span_lengths(n_spans, len(words))

        # Select non-overlapping start positions
        total_span_length = sum(span_lengths)
        if total_span_length >= len(words):
            # Reduce span lengths
            scale = (len(words) - n_spans) / total_span_length
            span_lengths = [max(1, int(l * scale)) for l in span_lengths]

        # Find valid start positions
        starts = []
        occupied = set()

        for span_idx, span_len in enumerate(span_lengths):
            valid_starts = []
            for i in range(len(words) - span_len + 1):
                span_positions = set(range(i, i + span_len))
                if not span_positions & occupied:
                    valid_starts.append(i)

            if valid_starts:
                start = random.choice(valid_starts)
                starts.append((start, span_len, span_idx))
                occupied.update(range(start, start + span_len))
            else:
                break

        if not starts:
            return text, []

        # Sort by position
        starts.sort(key=lambda x: x[0])

        # Build masked text and collect original spans
        result_words = []
        original_spans = []
        prev_end = 0

        for start, span_len, span_idx in starts:
            # Add words before this span
            result_words.extend(words[prev_end:start])

            # Add sentinel token
            sentinel = self.sentinel_tokens[span_idx]
            result_words.append(sentinel)

            # Store original span
            original_spans.append(" ".join(words[start:start + span_len]))

            prev_end = start + span_len

        # Add remaining words
        result_words.extend(words[prev_end:])

        return " ".join(result_words), original_spans

    def _apply_prompt_template(self, text: str) -> str:
        """Apply prompt template to text."""
        input_cfg = self.config.input

        if input_cfg.prompt_template:
            return input_cfg.prompt_template.format(text=text)

        result = text
        if input_cfg.prefix:
            result = input_cfg.prefix + result
        if input_cfg.suffix:
            result = result + input_cfg.suffix

        return result

    def _prepare_text(self, text: str) -> Tuple[str, List[str]]:
        """Prepare text for inference."""
        input_cfg = self.config.input
        original_spans = []

        # Truncate if needed
        if input_cfg.truncate and len(text) > input_cfg.max_source_length:
            text = text[:input_cfg.max_source_length]

        # Check if text already has sentinel tokens
        has_sentinel = any(s in text for s in self.sentinel_tokens)

        # Mask random spans if configured and no sentinels present
        if input_cfg.mask_random_spans and not has_sentinel:
            text, original_spans = self._mask_random_spans(text)

        # Apply prompt template
        text = self._apply_prompt_template(text)

        return text, original_spans

    def _parse_output(self, output: str) -> Dict[str, str]:
        """Parse T5 output into sentinel -> infill mapping."""
        result = {}

        # Try to parse sentinel tokens from output
        for i, sentinel in enumerate(self.sentinel_tokens):
            if sentinel in output:
                # Find text after this sentinel
                parts = output.split(sentinel)
                if len(parts) > 1:
                    # Get text until next sentinel or end
                    infill = parts[1]
                    for next_sentinel in self.sentinel_tokens:
                        if next_sentinel in infill:
                            infill = infill.split(next_sentinel)[0]
                            break
                    result[sentinel] = infill.strip()

        return result

    def _reconstruct_text(self, input_text: str, infills: Dict[str, str]) -> str:
        """Reconstruct full text by replacing sentinels with infills."""
        result = input_text
        for sentinel, infill in infills.items():
            result = result.replace(sentinel, infill)
        return result

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Run infilling prediction on a single text."""
        prepared_text, original_spans = self._prepare_text(text)

        # Check if there are sentinels to fill
        has_sentinel = any(s in prepared_text for s in self.sentinel_tokens)
        if not has_sentinel:
            return []

        # Tokenize
        inputs = self.tokenizer(
            prepared_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.input.max_source_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        gen_kwargs = self._get_generation_config()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                output_scores=self.config.output.show_scores,
                return_dict_in_generate=True,
            )

        # Decode outputs
        results = []
        sequences = outputs.sequences

        for seq_idx in range(sequences.shape[0]):
            output_ids = sequences[seq_idx]
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

            # Clean up special tokens but keep sentinels
            clean_output = output_text.replace(self.tokenizer.pad_token or "", "")
            clean_output = clean_output.replace(self.tokenizer.eos_token or "", "")
            clean_output = clean_output.replace(self.tokenizer.bos_token or "", "")
            clean_output = clean_output.strip()

            # Parse infills
            infills = self._parse_output(clean_output)

            # Reconstruct text
            reconstructed = self._reconstruct_text(prepared_text, infills)

            result = {
                "input": prepared_text,
                "raw_output": clean_output,
                "infills": infills,
                "reconstructed": reconstructed,
            }

            if original_spans:
                result["original_spans"] = original_spans

            results.append(result)

        return results

    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Run infilling prediction on multiple texts."""
        all_results = []
        for text in texts:
            results = self.predict(text)
            all_results.append(results)
        return all_results

    def format_result(self, results: List[Dict[str, Any]],
                      entry_id: Optional[str] = None) -> str:
        """Format prediction results for display."""
        out_cfg = self.config.output

        if not results:
            return "No sentinels found in input"

        if out_cfg.output_format == OutputFormat.JSON:
            output = {"predictions": results}
            if entry_id:
                output["id"] = entry_id
            return json.dumps(output, indent=out_cfg.indent, ensure_ascii=False)

        elif out_cfg.output_format == OutputFormat.JSONL:
            output = {"predictions": results}
            if entry_id:
                output["id"] = entry_id
            return json.dumps(output, ensure_ascii=False)

        elif out_cfg.output_format == OutputFormat.CSV:
            rows = []
            for pred_idx, pred in enumerate(results):
                for sentinel, infill in pred.get("infills", {}).items():
                    row = {
                        "prediction_index": pred_idx,
                        "sentinel": sentinel,
                        "infill": infill,
                        "reconstructed": pred.get("reconstructed", ""),
                    }
                    if entry_id:
                        row["id"] = entry_id
                    rows.append(row)
            return rows

        else:  # TEXT format
            lines = []

            if entry_id:
                lines.append(f"ID: {entry_id}")

            for pred_idx, pred in enumerate(results):
                if len(results) > 1:
                    lines.append(f"\n--- Prediction {pred_idx + 1} ---")

                if out_cfg.show_input:
                    input_text = pred.get("input", "")
                    if out_cfg.colorize:
                        # Highlight sentinels in input
                        highlighted = input_text
                        for sentinel in self.sentinel_tokens:
                            highlighted = highlighted.replace(
                                sentinel,
                                f"{Colors.YELLOW}{Colors.BOLD}{sentinel}{Colors.RESET}"
                            )
                        lines.append(f"Input: {highlighted}")
                    else:
                        lines.append(f"Input: {input_text}")

                if out_cfg.show_raw_output:
                    lines.append(f"Raw output: {pred.get('raw_output', '')}")

                # Show infills
                infills = pred.get("infills", {})
                if infills:
                    lines.append("\nInfills:")
                    for sentinel, infill in infills.items():
                        if out_cfg.colorize:
                            lines.append(
                                f"  {Colors.YELLOW}{sentinel}{Colors.RESET} -> "
                                f"{Colors.GREEN}{Colors.BOLD}{infill}{Colors.RESET}"
                            )
                        else:
                            lines.append(f"  {sentinel} -> {infill}")

                if out_cfg.show_reconstructed:
                    reconstructed = pred.get("reconstructed", "")
                    if out_cfg.colorize:
                        # Highlight infilled portions
                        highlighted = reconstructed
                        for infill in infills.values():
                            if infill:
                                highlighted = highlighted.replace(
                                    infill,
                                    f"{Colors.GREEN}{Colors.BOLD}{infill}{Colors.RESET}",
                                    1  # Only replace first occurrence
                                )
                        lines.append(f"\nReconstructed: {highlighted}")
                    else:
                        lines.append(f"\nReconstructed: {reconstructed}")

                # Show original spans if available
                original_spans = pred.get("original_spans", [])
                if original_spans and out_cfg.verbose:
                    lines.append(f"\nOriginal spans: {original_spans}")

            return "\n".join(lines)


# =============================================================================
# CLI Argument Parser
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="T5/BART Text Infilling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic T5 usage with sentinel tokens
  %(prog)s --text "The <extra_id_0> sat on the <extra_id_1>."

  # Using FLAN-T5
  %(prog)s --model google/flan-t5-base --text "Fill: The <extra_id_0> is blue."

  # Using BART with mask token
  %(prog)s --model facebook/bart-base --model-family bart --text "The <mask> sat on the mat."

  # Random span masking
  %(prog)s --text "The quick brown fox jumps over the lazy dog" --mask-random --n-spans 2

  # With prompt prefix
  %(prog)s --text "The <extra_id_0> capital is Paris." --prefix "fill in the blank: "

  # Beam search with multiple outputs
  %(prog)s --text "The <extra_id_0> sat on the mat." --decoding beam_search --num-beams 5 --num-return 3

  # Process CSV file
  %(prog)s --input texts.csv --mask-random --n-spans 2

  # Save/load config
  %(prog)s --save-config my_config.json
  %(prog)s --config my_config.json --text "Test <extra_id_0>"
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
        default="t5-base",
        metavar="NAME",
        help="Model name or path (default: t5-base)"
    )
    model_group.add_argument(
        "--model-family",
        type=str,
        choices=[f.value for f in ModelFamily],
        default="t5",
        help="Model family for sentinel token detection (default: t5)"
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
    # Device & Precision
    # ===================
    device_group = parser.add_argument_group("Device & Precision")

    device_group.add_argument(
        "--device", "-d",
        type=str,
        choices=[d.value for d in DeviceType],
        default="auto",
        help="Device to use (default: auto)"
    )
    device_group.add_argument(
        "--device-map",
        type=str,
        help="Device map for model parallelism"
    )
    device_group.add_argument(
        "--precision",
        type=str,
        choices=[p.value for p in PrecisionType],
        default="fp32",
        help="Model precision (default: fp32)"
    )

    # ===================
    # Generation Configuration
    # ===================
    gen_group = parser.add_argument_group("Generation Configuration")

    gen_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )
    gen_group.add_argument(
        "--min-new-tokens",
        type=int,
        default=1,
        help="Minimum new tokens to generate (default: 1)"
    )
    gen_group.add_argument(
        "--max-length",
        type=int,
        help="Maximum total sequence length"
    )
    gen_group.add_argument(
        "--decoding", "--decoding-strategy",
        type=str,
        choices=[s.value for s in DecodingStrategy],
        default="greedy",
        help="Decoding strategy (default: greedy)"
    )
    gen_group.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)"
    )
    gen_group.add_argument(
        "--num-beam-groups",
        type=int,
        default=1,
        help="Number of beam groups for diverse beam search (default: 1)"
    )
    gen_group.add_argument(
        "--diversity-penalty",
        type=float,
        default=0.0,
        help="Diversity penalty for diverse beam search (default: 0.0)"
    )
    gen_group.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Stop beam search when all beams finish (default: True)"
    )
    gen_group.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )
    gen_group.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (for non-greedy decoding)"
    )
    gen_group.add_argument(
        "--temperature", "-t",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    gen_group.add_argument(
        "--top-k", "-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    gen_group.add_argument(
        "--top-p", "-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0)"
    )
    gen_group.add_argument(
        "--typical-p",
        type=float,
        default=1.0,
        help="Typical-p sampling parameter (default: 1.0)"
    )
    gen_group.add_argument(
        "--epsilon-cutoff",
        type=float,
        default=0.0,
        help="Epsilon cutoff for sampling (default: 0.0)"
    )
    gen_group.add_argument(
        "--eta-cutoff",
        type=float,
        default=0.0,
        help="Eta cutoff for sampling (default: 0.0)"
    )
    gen_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0)"
    )
    gen_group.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="N-gram size for no-repeat constraint (default: 0)"
    )
    gen_group.add_argument(
        "--encoder-repetition-penalty",
        type=float,
        default=1.0,
        help="Encoder repetition penalty (default: 1.0)"
    )
    gen_group.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty for beam search (default: 1.0)"
    )
    gen_group.add_argument(
        "--num-return", "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to return (default: 1)"
    )
    gen_group.add_argument(
        "--penalty-alpha",
        type=float,
        default=0.0,
        help="Penalty alpha for contrastive search (default: 0.0)"
    )

    # ===================
    # Input Configuration
    # ===================
    input_group = parser.add_argument_group("Input Configuration")

    input_group.add_argument(
        "--text",
        type=str,
        help="Direct text input with sentinel tokens"
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

    # Sentinel/masking options
    input_group.add_argument(
        "--sentinel-pattern",
        type=str,
        default="<extra_id_{i}>",
        help="Sentinel token pattern (default: <extra_id_{i}>)"
    )
    input_group.add_argument(
        "--mask-token",
        type=str,
        default="<mask>",
        help="Mask token for BART-style models (default: <mask>)"
    )
    input_group.add_argument(
        "--num-sentinel-tokens",
        type=int,
        default=100,
        help="Number of sentinel tokens to check (default: 100)"
    )
    input_group.add_argument(
        "--mask-random", "-r",
        action="store_true",
        help="Randomly mask spans in input text"
    )
    input_group.add_argument(
        "--n-spans", "-n",
        type=int,
        default=1,
        help="Number of spans to mask when using --mask-random (default: 1)"
    )
    input_group.add_argument(
        "--mean-span-length",
        type=float,
        default=3.0,
        help="Mean span length for masking (default: 3.0)"
    )
    input_group.add_argument(
        "--span-length-dist",
        type=str,
        default="geometric",
        choices=["geometric", "uniform", "fixed"],
        help="Span length distribution (default: geometric)"
    )
    input_group.add_argument(
        "--mask-pattern",
        type=str,
        metavar="REGEX",
        help="Only mask spans matching this regex pattern"
    )

    # Prompting
    input_group.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add before input text"
    )
    input_group.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add after input text"
    )
    input_group.add_argument(
        "--prompt-template",
        type=str,
        metavar="TEMPLATE",
        help="Prompt template with {text} placeholder"
    )

    # Input processing
    input_group.add_argument(
        "--max-source-length",
        type=int,
        default=512,
        help="Maximum source length in tokens (default: 512)"
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
        "--no-show-input",
        action="store_true",
        help="Don't show input in output"
    )
    output_group.add_argument(
        "--show-raw-output",
        action="store_true",
        help="Show raw model output"
    )
    output_group.add_argument(
        "--no-show-reconstructed",
        action="store_true",
        help="Don't show reconstructed text"
    )
    output_group.add_argument(
        "--show-scores",
        action="store_true",
        help="Show generation scores"
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

    # ===================
    # Batch processing
    # ===================
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )

    return parser


def args_to_config(args: argparse.Namespace) -> T5InfillingConfig:
    """Convert parsed arguments to T5InfillingConfig."""

    # Load base config from file if provided
    if args.config:
        config = T5InfillingConfig.from_json_file(args.config)
    else:
        config = T5InfillingConfig()

    # Override with CLI arguments
    # Model config
    config.model.model_name_or_path = args.model
    config.model.model_family = ModelFamily(args.model_family)
    config.model.revision = args.revision
    config.model.tokenizer_name_or_path = args.tokenizer
    config.model.use_fast_tokenizer = not args.no_fast_tokenizer
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
    config.inference.batch_size = args.batch_size

    # Generation params
    gen = config.inference.generation
    gen.max_new_tokens = args.max_new_tokens
    gen.min_new_tokens = args.min_new_tokens
    gen.max_length = args.max_length
    gen.decoding_strategy = DecodingStrategy(args.decoding)
    gen.num_beams = args.num_beams
    gen.num_beam_groups = args.num_beam_groups
    gen.diversity_penalty = args.diversity_penalty
    gen.early_stopping = not args.no_early_stopping
    gen.do_sample = args.do_sample
    gen.temperature = args.temperature
    gen.top_k = args.top_k
    gen.top_p = args.top_p
    gen.typical_p = args.typical_p
    gen.epsilon_cutoff = args.epsilon_cutoff
    gen.eta_cutoff = args.eta_cutoff
    gen.repetition_penalty = args.repetition_penalty
    gen.no_repeat_ngram_size = args.no_repeat_ngram_size
    gen.encoder_repetition_penalty = args.encoder_repetition_penalty
    gen.length_penalty = args.length_penalty
    gen.num_return_sequences = args.num_return
    gen.penalty_alpha = args.penalty_alpha

    # Input config
    config.input.text = args.text
    config.input.texts = args.texts
    config.input.input_file = args.input
    config.input.input_format = args.input_format
    config.input.text_column = args.text_column
    config.input.id_column = args.id_column
    config.input.sentinel_token_pattern = args.sentinel_pattern
    config.input.mask_token = args.mask_token
    config.input.num_sentinel_tokens = args.num_sentinel_tokens
    config.input.mask_random_spans = args.mask_random
    config.input.n_spans = args.n_spans
    config.input.mean_span_length = args.mean_span_length
    config.input.span_length_distribution = args.span_length_dist
    config.input.mask_pattern = args.mask_pattern
    config.input.prefix = args.prefix
    config.input.suffix = args.suffix
    config.input.prompt_template = args.prompt_template
    config.input.max_source_length = args.max_source_length
    config.input.truncate = not args.no_truncate
    config.input.skip_empty = args.skip_empty
    config.input.shuffle = args.shuffle
    config.input.sample_size = args.sample_size
    config.input.start_index = args.start_index
    config.input.end_index = args.end_index

    # Output config
    config.output.output_file = args.output
    config.output.output_format = OutputFormat(args.output_format)
    config.output.show_input = not args.no_show_input
    config.output.show_raw_output = args.show_raw_output
    config.output.show_reconstructed = not args.no_show_reconstructed
    config.output.show_scores = args.show_scores
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

def load_inputs(config: T5InfillingConfig) -> List[Dict[str, Any]]:
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
    pipe = T5InfillingPipeline(config)

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
            fieldnames = ["id", "prediction_index", "sentinel", "infill", "reconstructed"]
            csv_writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            csv_writer.writeheader()

    # Process entries
    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        entry_id = entry.get("id")

        results = pipe.predict(text)

        if not results:
            if config.output.verbose:
                print(f"[{i+1}/{len(entries)}] No sentinel tokens found in: {text[:50]}...")
            continue

        formatted = pipe.format_result(results, entry_id)

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
