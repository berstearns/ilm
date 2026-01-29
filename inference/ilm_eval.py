# -*- coding: utf-8 -*-
"""Non-Interactive Infilling Model Evaluation Script

Evaluates multiple infilling models (ILM, BERT/RoBERTa MLM, T5/BART) on accuracy metrics.

Metrics:
- Top-1 Accuracy (exact match)
- Unigram character overlap (recall + F1)
- Bigram character overlap (recall + F1)

Results are broken down by CEFR level and overall.

Usage:
    python ilm_eval.py -i data.csv --models ilm:../models/sto_ilm mlm:bert-base-uncased t5:t5-small
    python ilm_eval.py -i data.csv --models mlm:bert-base-uncased --print-every 50 --output results.json
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
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

# Try to import NLTK for human tokenization
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# =============================================================================
# Tokenization Strategies
# =============================================================================

class TokenInfo:
    """Information about a token including position and PoS tag."""
    def __init__(self, start: int, end: int, text: str, pos: str = 'UNK'):
        self.start = start
        self.end = end
        self.text = text
        self.pos = pos

    def to_tuple(self) -> Tuple[int, int, str]:
        """Return (start, end, text) tuple for backwards compatibility."""
        return (self.start, self.end, self.text)


def ensure_nltk_data():
    """Download required NLTK data if not present."""
    if not NLTK_AVAILABLE:
        return False
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading NLTK PoS tagger...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    return True


def get_word_positions_regex(text: str) -> List[TokenInfo]:
    """Get word positions using regex (original behavior)."""
    tokens = []
    for match in re.finditer(r'\b\w+\b', text):
        tokens.append(TokenInfo(match.start(), match.end(), match.group(), 'UNK'))
    return tokens


def get_word_positions_nltk(text: str) -> List[TokenInfo]:
    """Get word positions using NLTK with PoS tagging."""
    if not NLTK_AVAILABLE:
        return get_word_positions_regex(text)

    # Tokenize and get PoS tags
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    # Map tokens back to character positions
    tokens = []
    search_start = 0
    for word, pos in pos_tags:
        # Find the word in the text starting from search_start
        idx = text.find(word, search_start)
        if idx != -1:
            tokens.append(TokenInfo(idx, idx + len(word), word, pos))
            search_start = idx + len(word)

    return tokens


def select_mask_positions_with_pos(
    text: str,
    n_masks: int = 3,
    strategy: str = 'human-tokens'
) -> List[TokenInfo]:
    """
    Select positions to mask with PoS information.

    Args:
        text: Input text
        n_masks: Number of tokens to mask
        strategy: 'human-tokens' (NLTK) or 'regex-tokens' (simple regex)

    Returns:
        List of TokenInfo objects for selected positions
    """
    if strategy == 'human-tokens' and NLTK_AVAILABLE:
        word_positions = get_word_positions_nltk(text)
    else:
        word_positions = get_word_positions_regex(text)

    # Filter to only alphabetic tokens (skip punctuation, numbers)
    word_positions = [t for t in word_positions if t.text.isalpha()]

    if len(word_positions) < n_masks:
        n_masks = len(word_positions)
    if n_masks == 0:
        return []

    selected_indices = sorted(random.sample(range(len(word_positions)), n_masks))
    return [word_positions[i] for i in selected_indices]


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
        """Generate infill for masked text."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type identifier."""
        pass

    @abstractmethod
    def count_subtokens(self, word: str) -> int:
        """Count how many subtokens this word is tokenized into by this model's tokenizer."""
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

        try:
            ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, self.tokenizer)
        except ValueError:
            pass

        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(self.device)

    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        if not mask_positions:
            return text

        masked_text = self._apply_masks(text, mask_positions, " _")
        context_ids = ilm.tokenize_util.encode(masked_text, self.tokenizer)
        _blank_id = ilm.tokenize_util.encode(' _', self.tokenizer)[0]

        mask_types = ['word'] * len(mask_positions)
        for mask_type in mask_types:
            try:
                idx = context_ids.index(_blank_id)
                context_ids[idx] = self.additional_tokens_to_ids['<|infill_word|>']
            except ValueError:
                break

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
        result = []
        prev_end = 0
        for start, end, _ in sorted(positions, key=lambda x: x[0]):
            result.append(text[prev_end:start])
            result.append(mask_token)
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)

    def count_subtokens(self, word: str) -> int:
        """Count how many subtokens this word is tokenized into by ILM's GPT2 tokenizer."""
        if not word or not word.strip():
            return 1
        try:
            subtokens = ilm.tokenize_util.tokenize(word, self.tokenizer)
            return len(subtokens)
        except Exception:
            return 1


# =============================================================================
# MLM Model Wrapper (BERT, RoBERTa, etc.)
# =============================================================================

class MLMModelWrapper(BaseInfillingModel):
    """Wrapper for Masked Language Models (BERT, RoBERTa, DistilBERT, etc.)."""

    def __init__(self, model_name: str, device: torch.device):
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
            current_text = text
            sorted_positions = sorted(mask_positions, key=lambda x: x[0])

            for i, (start, end, orig_word) in enumerate(sorted_positions):
                words_before = len(re.findall(r'\b\w+\b', text[:start]))
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

        except Exception:
            return None

    def count_subtokens(self, word: str) -> int:
        """Count how many subtokens this word is tokenized into by the MLM tokenizer."""
        if not word or not word.strip():
            return 1
        try:
            tokens = self.tokenizer.tokenize(word)
            return len(tokens)
        except Exception:
            return 1


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
        self.is_bart = False

    @property
    def model_type(self) -> str:
        return "t5"

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        model_lower = self.model_name.lower()
        self.is_bart = any(x in model_lower for x in ['bart', 'mbart', 'pegasus'])

    def infill(self, text: str, mask_positions: List[Tuple[int, int, str]]) -> Optional[str]:
        if not mask_positions:
            return text

        if self.is_bart:
            masked_text = self._apply_masks_bart(text, mask_positions)
        else:
            masked_text = self._apply_masks_t5(text, mask_positions)

        try:
            inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=128, num_beams=1, do_sample=False)

            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            if self.is_bart:
                return self._reconstruct_bart(output_text)
            else:
                return self._reconstruct_t5(masked_text, output_text, mask_positions)

        except Exception:
            return None

    def _apply_masks_t5(self, text: str, positions: List[Tuple[int, int, str]]) -> str:
        result = []
        prev_end = 0
        for i, (start, end, _) in enumerate(sorted(positions, key=lambda x: x[0])):
            result.append(text[prev_end:start])
            result.append(f"<extra_id_{i}>")
            prev_end = end
        result.append(text[prev_end:])
        return ''.join(result)

    def _apply_masks_bart(self, text: str, positions: List[Tuple[int, int, str]]) -> str:
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
        infills = {}
        # Clean special tokens (check existence to avoid warnings)
        if self.tokenizer.pad_token:
            output = output.replace(self.tokenizer.pad_token, "")
        if self.tokenizer.eos_token:
            output = output.replace(self.tokenizer.eos_token, "")
        if getattr(self.tokenizer, 'bos_token', None):
            output = output.replace(self.tokenizer.bos_token, "")
        output = output.strip()

        for i in range(len(positions)):
            sentinel = f"<extra_id_{i}>"
            if sentinel in output:
                parts = output.split(sentinel)
                if len(parts) > 1:
                    infill = parts[1]
                    for j in range(i + 1, len(positions) + 1):
                        next_sentinel = f"<extra_id_{j}>"
                        if next_sentinel in infill:
                            infill = infill.split(next_sentinel)[0]
                            break
                    infills[sentinel] = infill.strip()

        result = masked_text
        for sentinel, infill in infills.items():
            result = result.replace(sentinel, infill)
        return result

    def _reconstruct_bart(self, output: str) -> str:
        # Clean special tokens (check existence to avoid warnings)
        if self.tokenizer.pad_token:
            output = output.replace(self.tokenizer.pad_token, "")
        if self.tokenizer.eos_token:
            output = output.replace(self.tokenizer.eos_token, "")
        if getattr(self.tokenizer, 'bos_token', None):
            output = output.replace(self.tokenizer.bos_token, "")
        return output.strip()

    def count_subtokens(self, word: str) -> int:
        """Count how many subtokens this word is tokenized into by the T5 tokenizer."""
        if not word or not word.strip():
            return 1
        try:
            tokens = self.tokenizer.tokenize(word)
            return len(tokens)
        except Exception:
            return 1


# =============================================================================
# Model Factory
# =============================================================================

def parse_model_spec(spec: str) -> Tuple[str, str]:
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
    if model_type == 'ilm':
        return ILMModelWrapper(model_path, device)
    elif model_type == 'mlm':
        return MLMModelWrapper(model_path, device)
    elif model_type == 't5':
        return T5ModelWrapper(model_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def discover_ilm_models(models_dir: str) -> List[str]:
    model_specs = []
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            path = os.path.join(models_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'additional_ids_to_tokens.pkl')):
                model_specs.append(f"ilm:{path}")
    return model_specs


# =============================================================================
# Utility Functions
# =============================================================================

def get_word_positions(text: str) -> List[Tuple[int, int, str]]:
    positions = []
    for match in re.finditer(r'\b\w+\b', text):
        positions.append((match.start(), match.end(), match.group()))
    return positions


def select_mask_positions(text: str, n_masks: int = 3) -> List[Tuple[int, int, str]]:
    word_positions = get_word_positions(text)
    if len(word_positions) < n_masks:
        n_masks = len(word_positions)
    if n_masks == 0:
        return []
    selected_indices = sorted(random.sample(range(len(word_positions)), n_masks))
    return [word_positions[i] for i in selected_indices]


def truncate_text(text: str, max_chars: int = 500) -> str:
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
    if not mask_positions or not result_text:
        return []

    orig_tokens = list(re.finditer(r'\b\w+\b', original_text))
    result_tokens = list(re.finditer(r'\b\w+\b', result_text))

    if not orig_tokens or not result_tokens:
        return [''] * len(mask_positions)

    orig_word_positions = [(m.start(), m.end(), m.group()) for m in orig_tokens]

    masked_indices = []
    for start, end, word in mask_positions:
        for i, (ws, we, w) in enumerate(orig_word_positions):
            if ws == start and w == word:
                masked_indices.append(i)
                break

    infilled = []
    result_word_list = [m.group() for m in result_tokens]

    for idx in masked_indices:
        if idx < len(result_word_list):
            infilled.append(result_word_list[idx])
        else:
            infilled.append('')

    return infilled


# =============================================================================
# Metric Calculations
# =============================================================================

def calc_accuracy(pred: str, orig: str) -> float:
    """Calculate exact match accuracy (case-insensitive)."""
    return 1.0 if pred.lower() == orig.lower() else 0.0


def get_char_ngrams(word: str, n: int) -> set:
    """Get character n-grams from a word."""
    word = word.lower()
    if n == 1:
        return set(word)
    return set(word[i:i+n] for i in range(len(word) - n + 1)) if len(word) >= n else set()


def calc_char_overlap(pred: str, orig: str, n: int = 1) -> Dict[str, float]:
    """
    Calculate character n-gram overlap metrics.
    Returns dict with 'recall', 'precision', 'f1'.
    """
    pred_ngrams = get_char_ngrams(pred, n)
    orig_ngrams = get_char_ngrams(orig, n)

    if not orig_ngrams:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}

    intersection = pred_ngrams & orig_ngrams

    recall = len(intersection) / len(orig_ngrams) if orig_ngrams else 0.0
    precision = len(intersection) / len(pred_ngrams) if pred_ngrams else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'recall': recall, 'precision': precision, 'f1': f1}


# =============================================================================
# Metrics Accumulator
# =============================================================================

class MetricsAccumulator:
    """Accumulates metrics across predictions with CEFR and PoS breakdown."""

    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

        # Initialize accumulators
        # Structure: {model_name: {metric: [values]}}
        self.overall = {name: self._empty_metrics() for name in model_names}
        self.by_cefr = {
            level: {name: self._empty_metrics() for name in model_names}
            for level in self.cefr_levels
        }
        # PoS breakdown - dynamically populated
        # Structure: {pos_tag: {model_name: {metric: [values]}}}
        self.by_pos = {}
        self.pos_tags_seen = set()
        # Subtoken count breakdown - dynamically populated per model
        # Structure: {model_name: {subtoken_count: {metric: [values]}}}
        self.by_n_llm_subtokens = {}
        self.n_llm_subtokens_seen = set()

    def _empty_metrics(self) -> Dict[str, List[float]]:
        return {
            'accuracy': [],
            'unigram_recall': [],
            'unigram_precision': [],
            'unigram_f1': [],
            'bigram_recall': [],
            'bigram_precision': [],
            'bigram_f1': [],
        }

    def add(self, model_name: str, pred: str, orig: str, cefr: str, pos: str = 'UNK', n_llm_subtokens: Optional[int] = None):
        """Add a single prediction result."""
        # Calculate metrics
        acc = calc_accuracy(pred, orig)
        uni = calc_char_overlap(pred, orig, n=1)
        bi = calc_char_overlap(pred, orig, n=2)

        metrics = {
            'accuracy': acc,
            'unigram_recall': uni['recall'],
            'unigram_precision': uni['precision'],
            'unigram_f1': uni['f1'],
            'bigram_recall': bi['recall'],
            'bigram_precision': bi['precision'],
            'bigram_f1': bi['f1'],
        }

        # Add to overall
        for metric, value in metrics.items():
            self.overall[model_name][metric].append(value)

        # Add to CEFR-specific
        cefr_upper = cefr.upper() if cefr else ''
        if cefr_upper in self.cefr_levels:
            for metric, value in metrics.items():
                self.by_cefr[cefr_upper][model_name][metric].append(value)

        # Add to PoS-specific
        if pos and pos != 'UNK':
            if pos not in self.by_pos:
                self.by_pos[pos] = {name: self._empty_metrics() for name in self.model_names}
                self.pos_tags_seen.add(pos)
            for metric, value in metrics.items():
                self.by_pos[pos][model_name][metric].append(value)

        # Add to subtoken count-specific (per-model breakdown)
        if n_llm_subtokens is not None:
            # Initialize per-model storage if needed
            if model_name not in self.by_n_llm_subtokens:
                self.by_n_llm_subtokens[model_name] = {}
            # Create bucket for this subtoken count if needed
            n_str = str(n_llm_subtokens)
            if n_str not in self.by_n_llm_subtokens[model_name]:
                self.by_n_llm_subtokens[model_name][n_str] = self._empty_metrics()
                self.n_llm_subtokens_seen.add(n_llm_subtokens)
            # Append metrics
            for metric, value in metrics.items():
                self.by_n_llm_subtokens[model_name][n_str][metric].append(value)

    def get_summary(self, metrics_dict: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics from accumulated values."""
        summary = {}
        for model_name, metrics in metrics_dict.items():
            model_summary = {}
            for metric, values in metrics.items():
                if values:
                    model_summary[metric] = sum(values) / len(values)
                else:
                    model_summary[metric] = 0.0
            model_summary['n_predictions'] = len(metrics['accuracy'])
            model_summary['n_correct'] = int(sum(metrics['accuracy']))
            summary[model_name] = model_summary
        return summary

    def get_overall_summary(self) -> Dict[str, Dict[str, float]]:
        """Get overall metrics summary."""
        return self.get_summary(self.overall)

    def get_cefr_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get per-CEFR metrics summary."""
        return {level: self.get_summary(metrics) for level, metrics in self.by_cefr.items()}

    def get_pos_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get per-PoS metrics summary."""
        return {pos: self.get_summary(metrics) for pos, metrics in self.by_pos.items()}

    def _bucket_subtoken_count(self, n: int) -> str:
        """Bucket a subtoken count into ranges: '1', '2', '3-4', '5+'."""
        if n == 1:
            return '1'
        elif n == 2:
            return '2'
        elif n in (3, 4):
            return '3-4'
        else:  # n >= 5
            return '5+'

    def _get_bucketed_subtoken_breakdown(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Convert exact subtoken counts to bucketed breakdown."""
        bucketed = {}
        for model_name, subtoken_dict in self.by_n_llm_subtokens.items():
            bucketed[model_name] = {}
            for n_str, metrics in subtoken_dict.items():
                n = int(n_str)
                bucket = self._bucket_subtoken_count(n)
                if bucket not in bucketed[model_name]:
                    bucketed[model_name][bucket] = self._empty_metrics()
                # Append all metric values to the bucket
                for metric, values in metrics.items():
                    bucketed[model_name][bucket][metric].extend(values)
        return bucketed

    def get_n_llm_subtokens_summary(self, bucketed: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get per-subtoken-count metrics summary.

        Args:
            bucketed: If True, use bucketed ranges (1, 2, 3-4, 5+). If False, use exact counts.

        Returns:
            Structure: {model_name: {subtoken_count/bucket: {metric: value}}}
        """
        if bucketed:
            bucketed_breakdown = self._get_bucketed_subtoken_breakdown()
            result = {}
            for model_name, bucket_dict in bucketed_breakdown.items():
                result[model_name] = self.get_summary(bucket_dict)
            return result
        else:
            # Exact counts: return summary for each model's subtoken breakdown
            result = {}
            for model_name, subtoken_dict in self.by_n_llm_subtokens.items():
                result[model_name] = self.get_summary(subtoken_dict)
            return result

    def get_full_results(self, subtoken_granularity: str = 'exact') -> Dict[str, Any]:
        """Get complete results dictionary.

        Args:
            subtoken_granularity: 'exact', 'bucketed', or 'both'

        Returns:
            Dictionary with 'overall', 'by_cefr', 'by_pos', and optionally 'by_n_llm_subtokens'
        """
        results = {
            'overall': self.get_overall_summary(),
            'by_cefr': self.get_cefr_summary(),
            'by_pos': self.get_pos_summary(),
        }

        if subtoken_granularity in ('exact', 'both'):
            results['by_n_llm_subtokens'] = self.get_n_llm_subtokens_summary(bucketed=False)

        if subtoken_granularity in ('bucketed', 'both'):
            results['by_n_llm_subtokens_bucketed'] = self.get_n_llm_subtokens_summary(bucketed=True)

        return results

    def print_progress(self, n_processed: int, n_total: int):
        """Print current progress metrics."""
        summary = self.get_overall_summary()
        print(f"\n[{n_processed}/{n_total}] Metrics so far:")
        for model_name in self.model_names:
            m = summary[model_name]
            print(f"  {model_name:20s} | acc: {m['accuracy']:.3f} | "
                  f"uni_r: {m['unigram_recall']:.3f} | uni_f1: {m['unigram_f1']:.3f} | "
                  f"bi_r: {m['bigram_recall']:.3f} | bi_f1: {m['bigram_f1']:.3f}")


# =============================================================================
# Config Builder
# =============================================================================

def build_config(args, model_specs: List[str], model_labels: List[str],
                 n_texts: int, n_processed: int, status: str,
                 start_time: datetime, device: str) -> Dict[str, Any]:
    """Build comprehensive config dict with all parameters."""
    import transformers

    elapsed = (datetime.now() - start_time).total_seconds()

    return {
        # Status
        'status': status,
        'n_texts_total': n_texts,
        'n_texts_processed': n_processed,
        'progress_pct': round(100 * n_processed / n_texts, 2) if n_texts > 0 else 0,

        # Timestamps
        'started_at': start_time.isoformat(),
        'updated_at': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 2),
        'texts_per_second': round(n_processed / elapsed, 3) if elapsed > 0 else 0,

        # CLI Arguments
        'args': {
            'input': args.input,
            'output': args.output,
            'models': args.models,
            'models_dir': args.models_dir,
            'n_masks': args.n_masks,
            'samples_per_text': args.samples_per_text,
            'max_chars': args.max_chars,
            'limit': args.limit,
            'seed': args.seed,
            'print_every': args.print_every,
            'quiet': args.quiet,
            'masking': args.masking,
            'subtoken_granularity': args.subtoken_granularity,
        },

        # Models
        'models': {
            'specs': model_specs,
            'labels': model_labels,
            'count': len(model_labels),
        },

        # Environment
        'environment': {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'transformers_version': transformers.__version__,
            'nltk_available': NLTK_AVAILABLE,
            'device': device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Non-Interactive Infilling Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate models on full dataset
  python ilm_eval.py -i data.csv --models ilm:../models/sto_ilm mlm:bert-base-uncased t5:t5-small

  # With progress every 50 texts and JSON output
  python ilm_eval.py -i data.csv --models mlm:bert-base-uncased --print-every 50 -o results.json

  # Limit to first 100 texts for quick test
  python ilm_eval.py -i data.csv --models mlm:bert-base-uncased --limit 100
"""
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input CSV file (required)')
    parser.add_argument('--models', type=str, nargs='+', default=[],
                        help='Model specifications (type:path)')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory containing ILM models (auto-discovers all)')
    parser.add_argument('--n-masks', type=int, default=3,
                        help='Number of tokens to mask (default: 3)')
    parser.add_argument('--max-chars', type=int, default=500,
                        help='Max characters per text (default: 500)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N texts (optional)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--print-every', type=int, default=100,
                        help='Print progress every N texts (default: 100, 0 to disable)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file for detailed results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--masking', type=str, default='human-tokens',
                        choices=['human-tokens', 'regex-tokens'],
                        help='Masking strategy: human-tokens (NLTK with PoS) or regex-tokens (default: human-tokens)')
    parser.add_argument('--samples-per-text', type=int, default=1,
                        help='Number of random mask samples per text (default: 1)')
    parser.add_argument('--subtoken-granularity', type=str, default='exact',
                        choices=['exact', 'bucketed', 'both'],
                        help='Subtoken count granularity: exact (individual counts), bucketed (ranges 1,2,3-4,5+), or both (default: exact)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Initialize NLTK if using human-tokens strategy
    if args.masking == 'human-tokens':
        if not NLTK_AVAILABLE:
            print("Warning: NLTK not available. Install with: pip install nltk")
            print("Falling back to regex-tokens strategy.")
            args.masking = 'regex-tokens'
        else:
            ensure_nltk_data()

    # Collect model specifications
    model_specs = list(args.models)
    if args.models_dir:
        discovered = discover_ilm_models(args.models_dir)
        model_specs = discovered + model_specs
        if discovered and not args.quiet:
            print(f"Auto-discovered {len(discovered)} ILM model(s) from {args.models_dir}")

    if not model_specs:
        print("Error: No models specified. Use --models or --models-dir")
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

    if not args.quiet:
        print(f"\nLoading {len(parsed_models)} model(s):")
        for i, (mtype, mpath, _) in enumerate(parsed_models):
            print(f"  [{i+1}] {mtype}:{mpath.split('/')[-1]}")
        print()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.quiet:
        print(f"Using device: {device}\n")

    # Load all models
    models = []
    model_labels = []
    for model_type, model_path, spec in parsed_models:
        try:
            if not args.quiet:
                print(f"Loading {model_type}:{model_path.split('/')[-1]}...")
            model = create_model(model_type, model_path, device)
            model.load()
            models.append(model)
            model_labels.append(f"{model.model_type}:{model.name}")
        except Exception as e:
            print(f"  Warning: Failed to load {spec}: {e}")

    if not models:
        print("Error: No models loaded successfully")
        return

    if not args.quiet:
        print(f"\nSuccessfully loaded {len(models)} model(s)")

    # Load CSV
    if not args.quiet:
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

    # Apply limit
    if args.limit:
        texts = texts[:args.limit]

    if not args.quiet:
        total_samples = len(texts) * args.samples_per_text
        print(f"Processing {len(texts)} texts × {args.samples_per_text} samples = {total_samples} total samples")
        print(f"  {args.n_masks} masks per sample")
        print("=" * 60)

    # Track start time and prepare config params
    start_time = datetime.now()
    device_str = str(device)

    # Initialize metrics accumulator
    accumulator = MetricsAccumulator(model_labels)

    # Process all texts
    sample_count = 0
    total_samples = len(texts) * args.samples_per_text

    for idx, entry in enumerate(texts):
        original_text = entry['text'].strip()
        cefr = entry['cefr']

        # Truncate if too long
        text = truncate_text(original_text, args.max_chars)

        # Run multiple samples per text
        for sample_idx in range(args.samples_per_text):
            sample_count += 1

            # Select positions to mask (with PoS if using human-tokens)
            # Each sample gets fresh random positions
            token_infos = select_mask_positions_with_pos(text, args.n_masks, args.masking)

            if not token_infos:
                continue

            # Convert to tuples for model compatibility and extract info
            mask_positions = [t.to_tuple() for t in token_infos]
            original_words = [t.text for t in token_infos]
            pos_tags = [t.pos for t in token_infos]

            # Generate infills from each model
            for model, label in zip(models, model_labels):
                try:
                    result = model.infill(text, mask_positions)
                    if result:
                        predicted_words = extract_infilled_words(text, result, mask_positions)

                        # Add each prediction to accumulator (with PoS tag and subtoken count)
                        for pred, orig, pos in zip(predicted_words, original_words, pos_tags):
                            if pred:  # Skip empty predictions
                                # Count subtokens using this model's tokenizer
                                n_subtokens = model.count_subtokens(orig)
                                accumulator.add(label, pred, orig, cefr, pos, n_llm_subtokens=n_subtokens)
                except Exception:
                    pass  # Skip failed predictions

        # Print progress and save partial results (after all samples for this text)
        if not args.quiet and args.print_every > 0 and (idx + 1) % args.print_every == 0:
            accumulator.print_progress(idx + 1, len(texts))
            # Save partial results if output file specified
            if args.output:
                partial_results = accumulator.get_full_results(subtoken_granularity=args.subtoken_granularity)
                partial_results['config'] = build_config(
                    args=args,
                    model_specs=model_specs,
                    model_labels=model_labels,
                    n_texts=len(texts),
                    n_processed=idx + 1,
                    status='partial',
                    start_time=start_time,
                    device=device_str,
                )
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(partial_results, f, indent=2, ensure_ascii=False)

    # Final results
    results = accumulator.get_full_results(subtoken_granularity=args.subtoken_granularity)
    results['config'] = build_config(
        args=args,
        model_specs=model_specs,
        model_labels=model_labels,
        n_texts=len(texts),
        n_processed=len(texts),
        status='complete',
        start_time=start_time,
        device=device_str,
    )

    # Print final summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        print("\nOVERALL:")
        for model_name in model_labels:
            m = results['overall'][model_name]
            print(f"  {model_name:20s}")
            print(f"    Accuracy:        {m['accuracy']:.4f} ({m['n_correct']}/{m['n_predictions']})")
            print(f"    Unigram Recall:  {m['unigram_recall']:.4f}")
            print(f"    Unigram F1:      {m['unigram_f1']:.4f}")
            print(f"    Bigram Recall:   {m['bigram_recall']:.4f}")
            print(f"    Bigram F1:       {m['bigram_f1']:.4f}")

        print("\nBY CEFR LEVEL:")
        for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
            level_data = results['by_cefr'][level]
            has_data = any(level_data[m]['n_predictions'] > 0 for m in model_labels)
            if has_data:
                print(f"\n  {level}:")
                for model_name in model_labels:
                    m = level_data[model_name]
                    if m['n_predictions'] > 0:
                        print(f"    {model_name:20s} | acc: {m['accuracy']:.3f} | "
                              f"uni_f1: {m['unigram_f1']:.3f} | bi_f1: {m['bigram_f1']:.3f} | "
                              f"n={m['n_predictions']}")

    # Save JSON output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")

    # Print compact results dict
    print("\n" + "=" * 60)
    print("RESULTS DICT:")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
