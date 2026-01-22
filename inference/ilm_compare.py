# -*- coding: utf-8 -*-
"""Multi-Model Infilling Comparison Script

Loads multiple infilling models (ILM, BERT/RoBERTa MLM, T5/BART) and shows
side-by-side predictions for comparison.

Usage:
    # Compare ILM with BERT and T5
    python ilm_compare.py -i data.csv --models ilm:../models/sto_ilm mlm:bert-base-uncased t5:t5-base

    # Compare multiple MLM models
    python ilm_compare.py -i data.csv --models mlm:bert-base-uncased mlm:roberta-base mlm:distilbert-base-uncased

    # Auto-discover ILM models and add HF models
    python ilm_compare.py -i data.csv --models-dir ../models --models mlm:bert-base-uncased
"""

import argparse
import csv
import os
import pickle
import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import (
    GPT2LMHeadModel,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

# Try to import ILM modules (may not be available)
try:
    import ilm.tokenize_util
    from ilm.infer import infill_with_ilm
    ILM_AVAILABLE = True
except ImportError:
    ILM_AVAILABLE = False


# =============================================================================
# ANSI Colors
# =============================================================================

COLORS = [
    '\033[92m',  # Green
    '\033[94m',  # Blue
    '\033[93m',  # Yellow
    '\033[95m',  # Magenta
    '\033[96m',  # Cyan
    '\033[91m',  # Red
]
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'


def get_color(idx):
    """Get a color for model index."""
    return COLORS[idx % len(COLORS)]


# =============================================================================
# Abstract Base Model
# =============================================================================

class BaseInfillingModel(ABC):
    """Abstract base class for infilling models."""

    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device

    @abstractmethod
    def load(self):
        """Load the model."""
        pass

    @abstractmethod
    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        """
        Generate infill for masked text.

        Args:
            text: Original text (without masks)
            mask_positions: List of (start, end, original_word) tuples

        Returns:
            Reconstructed text with infills, or None on failure
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier."""
        pass


# =============================================================================
# ILM Model Wrapper
# =============================================================================

class ILMModelWrapper(BaseInfillingModel):
    """Wrapper for ILM (Infilling Language Model) models."""

    def __init__(self, model_dir: str, device: torch.device):
        name = os.path.basename(model_dir)
        super().__init__(name, device)
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.additional_tokens_to_ids = None

    @property
    def model_type(self) -> str:
        return "ilm"

    def load(self):
        if not ILM_AVAILABLE:
            raise RuntimeError("ILM module not available. Install ilm package.")

        additional_ids_to_tokens_path = os.path.join(
            self.model_dir, 'additional_ids_to_tokens.pkl'
        )
        with open(additional_ids_to_tokens_path, 'rb') as f:
            additional_ids_to_tokens = pickle.load(f)
        self.additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}

        self.tokenizer = ilm.tokenize_util.Tokenizer.GPT2

        # Update tokenizer with special tokens
        try:
            ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, self.tokenizer)
        except ValueError:
            pass  # Already updated

        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(self.device)

    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        if not mask_positions:
            return text

        # Build masked text with ILM-style blanks
        masked_text = self._apply_masks(text, mask_positions, " _")

        # Tokenize
        context_ids = ilm.tokenize_util.encode(masked_text, self.tokenizer)

        # Get blank token id
        _blank_id = ilm.tokenize_util.encode(' _', self.tokenizer)[0]

        # Replace blanks with infill tokens
        mask_types = ['word'] * len(mask_positions)
        for mask_type in mask_types:
            try:
                idx = context_ids.index(_blank_id)
                context_ids[idx] = self.additional_tokens_to_ids['<|infill_word|>']
            except ValueError:
                break

        # Generate
        generated = infill_with_ilm(
            self.model,
            self.additional_tokens_to_ids,
            context_ids,
            num_infills=1
        )

        if generated:
            return ilm.tokenize_util.decode(generated[0], self.tokenizer)
        return None

    def _apply_masks(self, text: str, positions: List[Tuple[int, int, str]], mask_token: str) -> str:
        """Apply mask tokens at specified positions."""
        result = []
        prev_end = 0
        for start, end, _ in sorted(positions, key=lambda x: x[0]):
            result.append(text[prev_end:start])
            result.append(mask_token)
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)


# =============================================================================
# MLM Model Wrapper (BERT, RoBERTa, etc.)
# =============================================================================

class MLMModelWrapper(BaseInfillingModel):
    """Wrapper for Masked Language Models (BERT, RoBERTa, DistilBERT, etc.)."""

    def __init__(self, model_name: str, device: torch.device):
        # Create display name from model path
        display_name = model_name.split('/')[-1]
        super().__init__(display_name, device)
        self.model_name = model_name
        self.pipe = None
        self.tokenizer = None
        self.mask_token = None

    @property
    def model_type(self) -> str:
        return "mlm"

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.mask_token = self.tokenizer.mask_token

        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        device_arg = 0 if self.device.type == 'cuda' else -1 if self.device.type == 'cpu' else self.device.type
        self.pipe = pipeline("fill-mask", model=model, tokenizer=self.tokenizer, device=device_arg, top_k=1)

    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        if not mask_positions:
            return text

        try:
            # Fill masks one at a time (HF pipeline doesn't support multiple masks)
            current_text = text
            sorted_positions = sorted(mask_positions, key=lambda x: x[0])

            for i, (start, end, orig_word) in enumerate(sorted_positions):
                # Find the current position of this word in the evolving text
                # We need to rebuild with single mask each time
                words_before = len(re.findall(r'\b\w+\b', text[:start]))

                # Tokenize current text and replace the target word with mask
                current_words = list(re.finditer(r'\b\w+\b', current_text))
                if words_before < len(current_words):
                    match = current_words[words_before]
                    masked_text = current_text[:match.start()] + self.mask_token + current_text[match.end():]

                    results = self.pipe(masked_text)
                    if results:
                        if isinstance(results[0], list):
                            results = results[0]
                        current_text = results[0]['sequence']

            return current_text

        except Exception as e:
            return None

    def _apply_masks(self, text: str, positions: List[Tuple[int, int, str]], mask_token: str) -> str:
        """Apply mask tokens at specified positions."""
        result = []
        prev_end = 0
        for start, end, _ in sorted(positions, key=lambda x: x[0]):
            result.append(text[prev_end:start])
            result.append(mask_token)
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)


# =============================================================================
# T5/BART Model Wrapper
# =============================================================================

class T5ModelWrapper(BaseInfillingModel):
    """Wrapper for T5/BART seq2seq infilling models."""

    def __init__(self, model_name: str, device: torch.device):
        display_name = model_name.split('/')[-1]
        super().__init__(display_name, device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.sentinel_pattern = "<extra_id_{i}>"
        self.is_bart = False

    @property
    def model_type(self) -> str:
        return "t5"

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        # Check if BART-style model
        model_lower = self.model_name.lower()
        self.is_bart = any(x in model_lower for x in ['bart', 'mbart', 'pegasus'])

    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        if not mask_positions:
            return text

        # Build masked text with sentinel tokens
        if self.is_bart:
            masked_text = self._apply_masks_bart(text, mask_positions)
        else:
            masked_text = self._apply_masks_t5(text, mask_positions)

        try:
            # Tokenize and generate
            inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=1,
                    do_sample=False,
                )

            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Clean and reconstruct
            if self.is_bart:
                return self._reconstruct_bart(masked_text, output_text, mask_positions)
            else:
                return self._reconstruct_t5(masked_text, output_text, mask_positions)

        except Exception:
            return None

    def _apply_masks_t5(self, text: str, positions: List[Tuple[int, int, str]]) -> str:
        """Apply T5 sentinel tokens at specified positions."""
        result = []
        prev_end = 0
        for i, (start, end, _) in enumerate(sorted(positions, key=lambda x: x[0])):
            result.append(text[prev_end:start])
            result.append(f"<extra_id_{i}>")
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)

    def _apply_masks_bart(self, text: str, positions: List[Tuple[int, int, str]]) -> str:
        """Apply BART mask tokens at specified positions."""
        mask_token = self.tokenizer.mask_token or "<mask>"
        result = []
        prev_end = 0
        for start, end, _ in sorted(positions, key=lambda x: x[0]):
            result.append(text[prev_end:start])
            result.append(mask_token)
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)

    def _reconstruct_t5(self, masked_text: str, output: str, positions: List[Tuple[int, int, str]]) -> str:
        """Reconstruct text from T5 output."""
        # Parse sentinel tokens from output
        infills = {}

        # Clean output (check existence to avoid warnings)
        if self.tokenizer.pad_token:
            output = output.replace(self.tokenizer.pad_token, "")
        if self.tokenizer.eos_token:
            output = output.replace(self.tokenizer.eos_token, "")
        if getattr(self.tokenizer, 'bos_token', None):
            output = output.replace(self.tokenizer.bos_token, "")
        output = output.strip()

        # Extract infills
        for i in range(len(positions)):
            sentinel = f"<extra_id_{i}>"
            if sentinel in output:
                parts = output.split(sentinel)
                if len(parts) > 1:
                    infill = parts[1]
                    # Find next sentinel
                    for j in range(i + 1, len(positions) + 1):
                        next_sentinel = f"<extra_id_{j}>"
                        if next_sentinel in infill:
                            infill = infill.split(next_sentinel)[0]
                            break
                    infills[sentinel] = infill.strip()

        # Reconstruct
        result = masked_text
        for sentinel, infill in infills.items():
            result = result.replace(sentinel, infill)

        return result

    def _reconstruct_bart(self, masked_text: str, output: str, positions: List[Tuple[int, int, str]]) -> str:
        """Reconstruct text from BART output - BART outputs the full sequence."""
        # BART typically outputs the complete filled sequence (check existence to avoid warnings)
        if self.tokenizer.pad_token:
            output = output.replace(self.tokenizer.pad_token, "")
        if self.tokenizer.eos_token:
            output = output.replace(self.tokenizer.eos_token, "")
        if getattr(self.tokenizer, 'bos_token', None):
            output = output.replace(self.tokenizer.bos_token, "")
        return output.strip()


# =============================================================================
# Model Factory
# =============================================================================

def parse_model_spec(spec: str) -> Tuple[str, str]:
    """
    Parse model specification string.

    Format: type:model_name_or_path

    Examples:
        ilm:../models/sto_ilm
        mlm:bert-base-uncased
        mlm:roberta-base
        t5:t5-base
        t5:google/flan-t5-base
    """
    if ':' not in spec:
        raise ValueError(f"Invalid model spec '{spec}'. Expected format: type:model_name_or_path")

    model_type, model_path = spec.split(':', 1)
    model_type = model_type.lower()

    if model_type not in ('ilm', 'mlm', 't5', 'seq2seq'):
        raise ValueError(f"Unknown model type '{model_type}'. Expected: ilm, mlm, t5, seq2seq")

    if model_type == 'seq2seq':
        model_type = 't5'

    return model_type, model_path


def create_model(model_type: str, model_path: str, device: torch.device) -> BaseInfillingModel:
    """Create a model wrapper based on type."""
    if model_type == 'ilm':
        return ILMModelWrapper(model_path, device)
    elif model_type == 'mlm':
        return MLMModelWrapper(model_path, device)
    elif model_type == 't5':
        return T5ModelWrapper(model_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Utility Functions
# =============================================================================

def get_word_positions(text: str) -> List[Tuple[int, int, str]]:
    """Get positions of words in text (start, end, word)."""
    positions = []
    for match in re.finditer(r'\b\w+\b', text):
        positions.append((match.start(), match.end(), match.group()))
    return positions


def select_mask_positions(text: str, n_masks: int = 3) -> List[Tuple[int, int, str]]:
    """
    Select random word positions to mask.

    Returns: List of (start, end, original_word) tuples
    """
    word_positions = get_word_positions(text)

    if len(word_positions) < n_masks:
        n_masks = len(word_positions)

    if n_masks == 0:
        return []

    selected_indices = sorted(random.sample(range(len(word_positions)), n_masks))
    return [word_positions[i] for i in selected_indices]


def colorize_infills(original_text: str, result_text: str, mask_positions: List[Tuple[int, int, str]], color: str) -> str:
    """Colorize the infilled portions in the result text."""
    if not mask_positions or not result_text:
        return result_text or ""

    # Get word indices that were masked
    orig_tokens = list(re.finditer(r'\b\w+\b', original_text))
    orig_word_positions = [(m.start(), m.end(), m.group()) for m in orig_tokens]

    # Find which word indices were masked
    masked_indices = []
    for start, end, word in mask_positions:
        for i, (ws, we, w) in enumerate(orig_word_positions):
            if ws == start and w == word:
                masked_indices.append(i)
                break

    # Get result tokens
    result_tokens = list(re.finditer(r'\b\w+\b', result_text))

    if not result_tokens:
        return result_text

    # Build colored result by highlighting words at masked indices
    colored_parts = []
    prev_end = 0

    for i, match in enumerate(result_tokens):
        # Add text before this word
        colored_parts.append(result_text[prev_end:match.start()])

        # Colorize if this index was masked
        if i in masked_indices:
            colored_parts.append(f"{color}{BOLD}{match.group()}{RESET}")
        else:
            colored_parts.append(match.group())

        prev_end = match.end()

    # Add remaining text
    colored_parts.append(result_text[prev_end:])

    return ''.join(colored_parts)


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, trying to end at a sentence boundary."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')

    last_sentence_end = max(last_period, last_question, last_exclaim)

    if last_sentence_end > max_chars // 2:
        return text[:last_sentence_end + 1]

    last_space = truncated.rfind(' ')
    if last_space > 0:
        return text[:last_space] + "..."

    return truncated + "..."


def extract_infilled_words(original_text: str, result_text: str, mask_positions: List[Tuple[int, int, str]]) -> List[str]:
    """
    Extract the infilled words from a result by comparing with original.
    Returns list of infilled words in order of mask positions.
    """
    if not mask_positions or not result_text:
        return []

    # Tokenize both texts
    orig_tokens = list(re.finditer(r'\b\w+\b', original_text))
    result_tokens = list(re.finditer(r'\b\w+\b', result_text))

    if not orig_tokens or not result_tokens:
        return ['?'] * len(mask_positions)

    # Build position-to-index mapping for original text
    orig_word_positions = [(m.start(), m.end(), m.group()) for m in orig_tokens]

    # Find which indices were masked
    masked_indices = []
    for start, end, word in mask_positions:
        for i, (ws, we, w) in enumerate(orig_word_positions):
            if ws == start and w == word:
                masked_indices.append(i)
                break

    # Extract words at those indices from result
    infilled = []
    result_word_list = [m.group() for m in result_tokens]

    for idx in masked_indices:
        if idx < len(result_word_list):
            infilled.append(result_word_list[idx])
        else:
            infilled.append('?')

    return infilled


def format_markdown_table_multi(masked_words: List[str], model_results: Dict[str, List[str]]) -> str:
    """Format predictions from multiple models as a markdown table."""
    GREEN = '\033[92m'
    BOLD_LOCAL = '\033[1m'
    RESET_LOCAL = '\033[0m'

    # Header
    model_names = list(model_results.keys())
    headers = ['Position', 'Original'] + model_names
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator = '|' + '|'.join(['----------' for _ in headers]) + '|'

    lines = [header_line, separator]

    # Rows
    for i, orig_word in enumerate(masked_words):
        row = [str(i + 1), orig_word]
        for model_name in model_names:
            result_words = model_results[model_name]
            if i < len(result_words):
                pred = result_words[i]
                # Colorize prediction in green
                row.append(f"{GREEN}{BOLD_LOCAL}{pred}{RESET_LOCAL}")
            else:
                row.append('?')
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def discover_ilm_models(models_dir: str) -> List[str]:
    """Discover ILM model directories."""
    model_specs = []
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            path = os.path.join(models_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'additional_ids_to_tokens.pkl')):
                model_specs.append(f"ilm:{path}")
    return model_specs


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Infilling Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Specification Format:
  type:model_name_or_path

  Types:
    ilm   - ILM models (local directory with additional_ids_to_tokens.pkl)
    mlm   - Masked LM models (BERT, RoBERTa, DistilBERT, etc.)
    t5    - Seq2seq models (T5, FLAN-T5, BART, etc.)

Examples:
  # Compare different model types
  python ilm_compare.py -i data.csv \\
      --models ilm:../models/sto_ilm mlm:bert-base-uncased t5:t5-base

  # Compare multiple BERT variants
  python ilm_compare.py -i data.csv \\
      --models mlm:bert-base-uncased mlm:bert-large-uncased mlm:roberta-base

  # Auto-discover ILM models + add HuggingFace models
  python ilm_compare.py -i data.csv --models-dir ../models \\
      --models mlm:distilbert-base-uncased

  # Compare T5 variants
  python ilm_compare.py -i data.csv \\
      --models t5:t5-small t5:t5-base t5:google/flan-t5-base
"""
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input CSV file (required)')
    parser.add_argument('--models', type=str, nargs='+', default=[],
                        help='Model specifications (type:path, e.g., mlm:bert-base-uncased)')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory containing ILM models (auto-discovers all)')
    parser.add_argument('--n-masks', type=int, default=3,
                        help='Number of tokens to mask (default: 3)')
    parser.add_argument('--max-chars', type=int, default=500,
                        help='Max characters per text (default: 500)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in CSV (default: 0)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the texts randomly')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--markdown', '-md', action='store_true',
                        help='Output predictions in markdown table format')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Collect model specifications
    model_specs = list(args.models)

    # Auto-discover ILM models if directory specified
    if args.models_dir:
        discovered = discover_ilm_models(args.models_dir)
        model_specs = discovered + model_specs
        if discovered:
            print(f"Auto-discovered {len(discovered)} ILM model(s) from {args.models_dir}")

    if not model_specs:
        print("Error: No models specified. Use --models or --models-dir")
        print("\nExample: --models mlm:bert-base-uncased t5:t5-base")
        return

    # Parse model specs
    parsed_models = []
    for spec in model_specs:
        try:
            model_type, model_path = parse_model_spec(spec)
            parsed_models.append((model_type, model_path, spec))
        except ValueError as e:
            print(f"Error: {e}")
            return

    print(f"\nLoading {len(parsed_models)} model(s):")
    for i, (mtype, mpath, _) in enumerate(parsed_models):
        print(f"  [{i+1}] {get_color(i)}{mtype}:{mpath.split('/')[-1]}{RESET}")
    print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load all models
    models = []
    for model_type, model_path, spec in parsed_models:
        try:
            print(f"Loading {model_type}:{model_path.split('/')[-1]}...")
            model = create_model(model_type, model_path, device)
            model.load()
            models.append(model)
        except Exception as e:
            print(f"  Warning: Failed to load {spec}: {e}")

    if not models:
        print("Error: No models loaded successfully")
        return

    print(f"\nSuccessfully loaded {len(models)} model(s)")

    # Load CSV
    print(f"Loading texts from: {args.input}")
    texts = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append({
                'id': row.get('id', ''),
                'cefr': row.get('cefr_label', ''),
                'text': row.get('text', ''),
                'l1': row.get('l1', '')
            })

    if args.shuffle:
        random.shuffle(texts)

    print(f"Loaded {len(texts)} texts")
    print(f"Masking {args.n_masks} words per text")
    print("\nControls: [Enter] next | [r] re-mask current | [q] quit | [number] jump to index\n")
    print("=" * 80)

    idx = args.start_idx
    while 0 <= idx < len(texts):
        entry = texts[idx]
        original_text = entry['text'].strip()

        # Truncate if too long
        text = truncate_text(original_text, args.max_chars)

        # Select positions to mask (same for all models)
        mask_positions = select_mask_positions(text, args.n_masks)

        # Display info
        print(f"\n[{idx + 1}/{len(texts)}] ID: {entry['id'][:8] if entry['id'] else 'N/A'}... | CEFR: {entry['cefr']} | L1: {entry['l1']}")
        print("-" * 80)

        print("\nORIGINAL:")
        print(text)

        if mask_positions:
            masked_words = [word for _, _, word in mask_positions]
            # Show text with blanks
            display_masked = text
            offset = 0
            for start, end, word in mask_positions:
                adj_start = start + offset
                adj_end = end + offset
                replacement = f"{DIM}[___]{RESET}"
                display_masked = display_masked[:adj_start] + replacement + display_masked[adj_end:]
                offset += len(replacement) - (end - start)

            print("\nMASKED:")
            print(display_masked)
            print(f"\nMasked words: {masked_words}")

        # Generate infills from each model
        print("\nPREDICTIONS:")
        print("-" * 40)

        if args.markdown:
            # Collect results from all models for table format
            model_results = {}
            for i, model in enumerate(models):
                model_label = f"{model.model_type}:{model.name}"
                try:
                    result = model.infill(text, mask_positions)
                    if result:
                        infilled = extract_infilled_words(text, result, mask_positions)
                        model_results[model_label] = infilled
                    else:
                        model_results[model_label] = ['—'] * len(mask_positions)
                except Exception as e:
                    model_results[model_label] = ['err'] * len(mask_positions)

            masked_words = [word for _, _, word in mask_positions]
            print(format_markdown_table_multi(masked_words, model_results))
        else:
            for i, model in enumerate(models):
                color = get_color(i)
                try:
                    result = model.infill(text, mask_positions)
                    if result:
                        # Colorize the infilled words
                        colored_result = colorize_infills(text, result, mask_positions, color)
                        print(f"\n{color}{BOLD}[{model.model_type}:{model.name}]{RESET}")
                        print(colored_result)
                    else:
                        print(f"\n{color}{BOLD}[{model.model_type}:{model.name}]{RESET}")
                        print(f"{DIM}(no output){RESET}")
                except Exception as e:
                    print(f"\n{color}{BOLD}[{model.model_type}:{model.name}]{RESET}")
                    print(f"{DIM}Error: {e}{RESET}")

        print("\n" + "=" * 80)

        # Wait for user input
        try:
            user_input = input("\n[Enter=next, r=re-mask, q=quit, number=jump]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if user_input == 'q':
            print("Goodbye!")
            break
        elif user_input == 'r':
            continue
        elif user_input.isdigit():
            idx = int(user_input) - 1
            if idx < 0:
                idx = 0
            elif idx >= len(texts):
                idx = len(texts) - 1
        else:
            idx += 1

    print(f"\nProcessed texts. Final index: {idx + 1}")


if __name__ == '__main__':
    main()
