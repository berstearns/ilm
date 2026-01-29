# -*- coding: utf-8 -*-
"""Interactive ILM Inference Script

This script provides an interactive interface for evaluating Infill Language Models (ILMs).
It loads texts from a CSV file, randomly masks selected words, and generates predictions
using a trained ILM model.

## Overview

An Infill Language Model is a language model trained to fill in missing portions of text
given surrounding context. This script allows you to:

1. Load texts from a CSV file
2. Randomly mask words in the text
3. Generate multiple prediction candidates
4. View predictions with highlighted infills
5. Navigate through texts interactively

## Features

- **Interactive Navigation**: Move between texts, re-mask current text, or jump to specific indices
- **Flexible Masking**: Mask random words with configurable count
- **Multiple Predictions**: Generate multiple candidate infills for comparison
- **Color Highlighting**: Green highlighting for infilled portions in output
- **Markdown Output**: Optional markdown table format for predictions
- **Reproducibility**: Seed support for deterministic masking
- **Text Truncation**: Automatic truncation to prevent overly long texts

## CSV Input Format

The input CSV file should have the following columns:
- `id`: Unique identifier for the text
- `text`: The text content to process (required)
- `cefr_label`: Optional CEFR proficiency level (A1-C2)
- `l1`: Optional first language of author

Example CSV:
```
id,text,cefr_label,l1
001,"The quick brown fox jumps over the lazy dog.",B1,Spanish
002,"Machine learning is a subset of artificial intelligence.",B2,French
```

## Usage Examples

### Basic Usage
```bash
python ilm_interactive.py -i texts.csv --model-dir ../models/sto_ilm
```

### Advanced Options
```bash
# Generate 5 predictions per masked text, mask 4 words
python ilm_interactive.py -i texts.csv --n-masks 4 --num-infills 5

# Start from text 10, shuffle order, set random seed for reproducibility
python ilm_interactive.py -i texts.csv --start-idx 9 --shuffle --seed 42

# Output predictions as markdown table, limit text to 300 characters
python ilm_interactive.py -i texts.csv --markdown --max-chars 300
```

## Interactive Commands

While running, use these commands:
- `[Enter]`: Move to next text
- `r`: Re-mask the current text (generate new random masks)
- `q`: Quit the script
- `[number]`: Jump to text at index (1-indexed)

## How It Works

1. **Model Loading**: Loads a pre-trained GPT-2 based ILM with custom tokens
2. **Tokenization**: Uses GPT-2 tokenizer with additional special tokens for infill markers
3. **Masking**: Randomly selects words and replaces them with `_` marker
4. **Infilling**: Generates predictions using the ILM's bidirectional infill capability
5. **Colorization**: Highlights newly generated text in green
6. **Display**: Shows original, masked, and predicted versions

## Output Format

### Default Format
Shows full text with infilled portions highlighted in green:
```
--- Option 1 ---
The quick brown fox jumps over the lazy [GREEN]dog[RESET].
```

### Markdown Format (--markdown)
Displays a table comparing predictions:
```
| Position | Original | Option 1 | Option 2 |
|----------|----------|----------|----------|
| 1        | fox      | fox      | fox      |
| 2        | jumps    | jumps    | leaps    |
```

## Configuration

Key parameters to adjust:
- `--n-masks`: Number of words to mask per text (default: 3)
- `--num-infills`: Number of prediction candidates (default: 2)
- `--max-chars`: Maximum characters per text before truncation (default: 500)
- `--model-dir`: Path to trained ILM model directory
- `--seed`: Random seed for reproducible masking

## Requirements

- torch
- transformers (GPT2LMHeadModel)
- Python 3.6+

## Model Format

The model directory should contain:
- `pytorch_model.bin`: Pre-trained model weights
- `config.json`: Model configuration
- `additional_ids_to_tokens.pkl`: Pickle file mapping custom token IDs
"""

import argparse
import csv
import os
import pickle
import random
import re

import torch
from transformers import GPT2LMHeadModel

import ilm.tokenize_util
from ilm.infer import infill_with_ilm


# ANSI color codes
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'


def colorize_infills_by_comparison(masked_text, generated_text):
    """
    Colorize infilled portions by comparing masked text with generated output.

    Compares the masked template (with ' _' markers) against the generated text
    to identify which portions were newly infilled by the model. These portions
    are highlighted in green to distinguish them from original context.

    Args:
        masked_text (str): Original text with blanks marked as ' _'
        generated_text (str): Full text with infilled content from the model

    Returns:
        str: Generated text with infilled portions colored green, or original
             text if pattern matching fails

    Example:
        >>> masked = "The quick _ fox jumps"
        >>> generated = "The quick brown fox jumps"
        >>> result = colorize_infills_by_comparison(masked, generated)
        # Returns: "The quick [GREEN]brown[RESET] fox jumps"

    Note:
        Uses regex pattern matching with non-greedy matching to identify infills.
        Falls back to returning text as-is if regex matching fails.
    """
    # Split masked text by the blank marker to get context pieces
    blank_marker = ' _'
    context_pieces = masked_text.split(blank_marker)

    if len(context_pieces) <= 1:
        # No blanks found
        return generated_text

    # Build regex pattern to capture infilled portions
    # Escape special regex characters in context pieces
    escaped_pieces = [re.escape(piece) for piece in context_pieces]

    # Create pattern that captures text between context pieces
    # Use non-greedy matching for infills
    pattern_parts = []
    for i, piece in enumerate(escaped_pieces):
        pattern_parts.append(f'({piece})')
        if i < len(escaped_pieces) - 1:
            pattern_parts.append('(.+?)')  # Capture infill (non-greedy)

    pattern = ''.join(pattern_parts)

    try:
        match = re.match(pattern, generated_text, re.DOTALL)
        if match:
            # Reconstruct with colored infills
            result = []
            groups = match.groups()
            infill_idx = 0
            for i, group in enumerate(groups):
                if i % 2 == 0:
                    # Context piece (unchanged)
                    result.append(group)
                else:
                    # Infilled portion (colorize)
                    result.append(f'{GREEN}{BOLD}{group}{RESET}')
                    infill_idx += 1
            return ''.join(result)
    except re.error:
        pass

    # Fallback: return as-is if pattern matching fails
    return generated_text


def load_model_and_tokenizer(model_dir):
    """
    Load the ILM model and tokenizer from disk.

    Loads a GPT-2 based ILM model along with its custom tokenizer. The model
    must include additional special tokens for infill control (e.g., '<|infill_word|>').

    Args:
        model_dir (str): Path to model directory containing:
            - pytorch_model.bin: Model weights
            - config.json: Model configuration
            - additional_ids_to_tokens.pkl: Pickle file with token mappings

    Returns:
        tuple: (model, tokenizer, additional_tokens_to_ids, device)
            - model (GPT2LMHeadModel): Loaded model in eval mode
            - tokenizer: GPT-2 tokenizer
            - additional_tokens_to_ids (dict): Mapping of token names to IDs
            - device (torch.device): 'cuda' if available, else 'cpu'

    Raises:
        FileNotFoundError: If model files are not found
        Exception: If tokenizer update fails (caught and ignored if already updated)

    Example:
        >>> model, tok, token_map, device = load_model_and_tokenizer('../models/sto_ilm')
        >>> print(f"Model loaded on {device}")
        Model loaded on cuda
    """
    print("Loading tokenizer...")
    tokenizer = ilm.tokenize_util.Tokenizer.GPT2

    additional_ids_to_tokens_path = os.path.join(model_dir, 'additional_ids_to_tokens.pkl')
    with open(additional_ids_to_tokens_path, 'rb') as f:
        additional_ids_to_tokens = pickle.load(f)
    additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}

    try:
        ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
    except ValueError:
        pass  # Already updated

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    return model, tokenizer, additional_tokens_to_ids, device


def get_word_positions(text):
    """
    Extract positions of all words in text.

    Uses regex to find word boundaries and returns their character positions
    along with the word content.

    Args:
        text (str): Input text to analyze

    Returns:
        list: List of tuples (start_pos, end_pos, word_text)
            - start_pos (int): Character index where word starts
            - end_pos (int): Character index where word ends (exclusive)
            - word_text (str): The actual word

    Example:
        >>> get_word_positions("Hello world")
        [(0, 5, 'Hello'), (6, 11, 'world')]
    """
    positions = []
    for match in re.finditer(r'\b\w+\b', text):
        positions.append((match.start(), match.end(), match.group()))
    return positions


def mask_random_words(text, n_masks=3, mask_type='word'):
    """
    Randomly mask n words in the text by replacing them with ' _' marker.

    Selects n random words from the text and replaces them with ' _' (space
    followed by underscore). This creates a template for the model to perform
    infilling. Also tracks the original positions and words for later comparison.

    Args:
        text (str): Input text to mask
        n_masks (int): Number of words to mask (default: 3)
        mask_type (str): Type of mask ('word', 'sentence', 'ngram') - used for
                        infill token selection (default: 'word')

    Returns:
        tuple: (masked_text, mask_positions, mask_types)
            - masked_text (str): Text with randomly selected words replaced by ' _'
            - mask_positions (list): List of (position, original_word) tuples
            - mask_types (list): List of mask types for each masked word

    Example:
        >>> text = "The quick brown fox jumps"
        >>> masked, positions, types = mask_random_words(text, n_masks=2)
        >>> print(masked)
        "The _ brown _ jumps"
        >>> print(positions)
        [(4, 'quick'), (14, 'fox')]

    Note:
        - If n_masks > number of words in text, masks all available words
        - Adjusts positions for text modifications as masks are applied
        - Returns empty lists if text has no words
    """
    word_positions = get_word_positions(text)

    if len(word_positions) < n_masks:
        n_masks = len(word_positions)

    if n_masks == 0:
        return text, [], []

    # Select random word positions to mask
    selected_indices = sorted(random.sample(range(len(word_positions)), n_masks))

    # Build masked text by replacing words with _
    masked_text = text
    offset = 0
    mask_positions = []
    mask_types = []

    for idx in selected_indices:
        start, end, word = word_positions[idx]
        adjusted_start = start + offset
        adjusted_end = end + offset

        # Replace word with " _" (space + underscore for tokenization)
        replacement = " _"
        masked_text = masked_text[:adjusted_start] + replacement + masked_text[adjusted_end:]

        offset += len(replacement) - (end - start)
        mask_positions.append((adjusted_start, word))
        mask_types.append(mask_type)

    return masked_text, mask_positions, mask_types


def infill_text(model, tokenizer, additional_tokens_to_ids, text, mask_types, num_infills=2):
    """
    Generate multiple infill predictions for masked text.

    Takes a template with ' _' markers and generates multiple candidate infills
    using the trained ILM model. The mask_types parameter controls which infill
    token is used for each blank.

    Args:
        model (GPT2LMHeadModel): Trained ILM model
        tokenizer: GPT-2 tokenizer with custom tokens
        additional_tokens_to_ids (dict): Mapping of token names to IDs
        text (str): Masked text template with ' _' markers
        mask_types (list): List of mask types ('word', 'sentence', 'ngram') for each blank
        num_infills (int): Number of prediction candidates to generate (default: 2)

    Returns:
        list: List of num_infills decoded text strings with infilled content,
              with infilled portions highlighted in green

    Example:
        >>> masked_text = "The _ brown fox"
        >>> results = infill_text(model, tok, token_map, masked_text, ['word'], num_infills=3)
        >>> for i, result in enumerate(results, 1):
        ...     print(f"Option {i}: {result}")
        Option 1: The [GREEN]quick[RESET] brown fox
        Option 2: The [GREEN]big[RESET] brown fox
        Option 3: The [GREEN]fat[RESET] brown fox

    Raises:
        ValueError: If blank token not found in tokenized text
        Exception: Propagates any errors from the underlying infill_with_ilm function

    Note:
        - Replaces blanks with appropriate infill tokens based on mask_types
        - Colorizes infilled portions green for visual distinction
        - Handles edge cases like missing blanks gracefully
    """
    original_masked_text = text  # Keep for colorization

    # Tokenize
    context_ids = ilm.tokenize_util.encode(text, tokenizer)

    # Get blank token id
    _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]

    # Replace blanks with infill tokens
    infill_token_map = {
        'word': '<|infill_word|>',
        'sentence': '<|infill_sentence|>',
        'ngram': '<|infill_ngram|>',
    }

    for mask_type in mask_types:
        try:
            idx = context_ids.index(_blank_id)
            token_name = infill_token_map.get(mask_type, '<|infill_word|>')
            context_ids[idx] = additional_tokens_to_ids[token_name]
        except ValueError:
            break  # No more blanks

    # Generate infills
    generated = infill_with_ilm(
        model,
        additional_tokens_to_ids,
        context_ids,
        num_infills=num_infills
    )

    # Decode results and colorize infilled portions
    results = []
    for g in generated:
        decoded = ilm.tokenize_util.decode(g, tokenizer)
        colorized = colorize_infills_by_comparison(original_masked_text, decoded)
        results.append(colorized)

    return results


def extract_infilled_words(original_text, result_text, mask_positions):
    """
    Extract infilled words by comparing original and result texts.

    Identifies which words replaced the masked positions by comparing the
    original text word positions with those in the result. Useful for
    structured output like markdown tables.

    Args:
        original_text (str): Original text with ' _' markers for blanks
        result_text (str): Full text with infilled content (ANSI codes removed)
        mask_positions (list): List of (position, original_word) tuples from masking

    Returns:
        list: List of infilled words in order of mask positions, or '?' if extraction fails

    Example:
        >>> original = "The _ brown _ jumps"
        >>> result = "The quick brown fox jumps"
        >>> positions = [(4, 'quick'), (14, 'fox')]
        >>> extract_infilled_words(original, result, positions)
        ['quick', 'fox']

    Note:
        - Parses both texts into word tokens using regex word boundary matching
        - Maps original mask positions to word indices
        - Returns '?' for any positions that fail to extract
        - Useful for creating structured output (markdown tables)
    """
    if not mask_positions or not result_text:
        return []

    # Get original words at mask positions
    original_words = [word for _, word in mask_positions]

    # Tokenize both texts
    import re
    orig_tokens = list(re.finditer(r'\b\w+\b', original_text))
    result_tokens = list(re.finditer(r'\b\w+\b', result_text))

    if not orig_tokens or not result_tokens:
        return ['?'] * len(mask_positions)

    # Build position-to-index mapping for original text
    orig_word_positions = [(m.start(), m.end(), m.group()) for m in orig_tokens]

    # Find which indices were masked
    masked_indices = []
    for start, word in mask_positions:
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


def format_markdown_table(masked_words, results, num_infills):
    """
    Format prediction results as a markdown table.

    Creates a structured table showing original masked words and their predicted
    replacements from each infill option, with predictions highlighted in green.

    Args:
        masked_words (list): List of original words that were masked
        results (list): List of lists containing extracted infilled words
                       results[i] is a list of infilled words for option i
        num_infills (int): Number of infill options (used for context)

    Returns:
        str: Markdown formatted table string ready for printing or saving

    Example:
        >>> masked_words = ['quick', 'fox']
        >>> results = [['speedy', 'dog'], ['fast', 'hound']]
        >>> table = format_markdown_table(masked_words, results, 2)
        >>> print(table)
        | Position | Original | Option 1 | Option 2 |
        |----------|----------|----------|----------|
        | 1        | quick    | speedy   | fast     |
        | 2        | fox      | dog      | hound    |

    Note:
        - Predictions are colored green with ANSI codes
        - Gracefully handles missing predictions with '?'
        - Header includes Position, Original, and Option N columns
    """
    # Header
    headers = ['Position', 'Original'] + [f'Option {i+1}' for i in range(len(results))]
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator = '|' + '|'.join(['----------' for _ in headers]) + '|'

    lines = [header_line, separator]

    # Rows
    for i, orig_word in enumerate(masked_words):
        row = [str(i + 1), orig_word]
        for result_words in results:
            if i < len(result_words):
                pred = result_words[i]
                # Colorize prediction in green
                row.append(f"{GREEN}{BOLD}{pred}{RESET}")
            else:
                row.append('?')
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def truncate_text(text, max_chars=500):
    """
    Truncate text intelligently at sentence or word boundaries.

    Attempts to truncate text at a natural boundary (sentence end) to maintain
    readability. Falls back to word boundaries if no sentence end is found,
    or to character boundary with ellipsis as last resort.

    Args:
        text (str): Text to truncate
        max_chars (int): Maximum character limit (default: 500)

    Returns:
        str: Truncated text (unchanged if already under max_chars), with "..."
             appended if truncation occurred

    Example:
        >>> long_text = "First sentence. Second sentence. Third sentence."
        >>> truncate_text(long_text, max_chars=20)
        "First sentence."

    Priority order:
        1. Sentence boundary (., ?, !) in second half of max_chars
        2. Word boundary (space) near max_chars
        3. Character boundary with ellipsis

    Note:
        - Returns text unchanged if already under max_chars
        - Requires sentence boundary to be at least halfway through max_chars
        - Useful for preventing extremely long texts in interactive mode
    """
    if len(text) <= max_chars:
        return text

    # Try to find a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')

    last_sentence_end = max(last_period, last_question, last_exclaim)

    if last_sentence_end > max_chars // 2:
        return text[:last_sentence_end + 1]

    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return text[:last_space] + "..."

    return truncated + "..."


def get_default_model_dir():
    """
    Get default model directory based on project structure.

    Assumes a standard project layout where the model is located at
    `../models/sto_ilm` relative to the script location.

    Returns:
        str: Path to default model directory

    Example:
        >>> model_dir = get_default_model_dir()
        >>> print(model_dir)
        /home/user/project/models/sto_ilm
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, 'models', 'sto_ilm')


def main():
    """
    Main entry point for the interactive ILM inference script.

    Parses command-line arguments, loads the model and data, and runs the
    interactive loop allowing users to navigate through texts and view
    predictions.

    Command-line Arguments:
        -i, --input (str, required): Path to input CSV file
        --n-masks (int): Number of tokens to mask (default: 3)
        --num-infills (int): Number of infill predictions (default: 2)
        --model-dir (str): Path to model directory (default: ../models/sto_ilm)
        --max-chars (int): Max characters per text (default: 500)
        --start-idx (int): Starting index in CSV (default: 0)
        --shuffle: Shuffle texts randomly (flag)
        --seed (int): Random seed for reproducibility
        --markdown, -md: Output as markdown table (flag)

    Interactive Controls:
        - [Enter]: Move to next text
        - 'r': Re-mask current text with new random selection
        - 'q': Quit script
        - [number]: Jump to specific text index (1-indexed)

    Workflow:
        1. Parse arguments and set random seed if provided
        2. Load model, tokenizer, and additional tokens
        3. Load texts from CSV
        4. Optionally shuffle texts
        5. Enter interactive loop:
           - Display text information (ID, CEFR, L1)
           - Show original text
           - Show masked version
           - Generate and display predictions
           - Wait for user input to continue

    Raises:
        FileNotFoundError: If input CSV or model files not found
        KeyboardInterrupt: Gracefully handles Ctrl+C exit
    """
    parser = argparse.ArgumentParser(
        description='Interactive ILM Inference',
        epilog="""
Examples:
  # Basic usage
  python ilm_interactive.py -i texts.csv

  # Generate 5 predictions, mask 4 words, shuffle texts
  python ilm_interactive.py -i texts.csv --n-masks 4 --num-infills 5 --shuffle

  # Start from text 10, use markdown output, set seed for reproducibility
  python ilm_interactive.py -i texts.csv --start-idx 9 --markdown --seed 42

  # Limit text length, use custom model directory
  python ilm_interactive.py -i texts.csv --max-chars 300 --model-dir /path/to/model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input CSV file (required)')
    parser.add_argument('--n-masks', type=int, default=3,
                        help='Number of tokens to mask (default: 3)')
    parser.add_argument('--num-infills', type=int, default=2,
                        help='Number of infill predictions to generate (default: 2)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Path to model directory (default: ../models/sto_ilm)')
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

    # Get model directory
    model_dir = args.model_dir if args.model_dir else get_default_model_dir()

    # Load model
    model, tokenizer, additional_tokens_to_ids, device = load_model_and_tokenizer(model_dir)

    # Load CSV
    print(f"\nLoading texts from: {args.input}")
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

        # Mask random words
        masked_text, mask_positions, mask_types = mask_random_words(text, args.n_masks)

        # Display info
        print(f"\n[{idx + 1}/{len(texts)}] ID: {entry['id'][:8]}... | CEFR: {entry['cefr']} | L1: {entry['l1']}")
        print("-" * 80)

        print("\n📝 ORIGINAL:")
        print(text)

        print("\n🎭 MASKED (words replaced with '_'):")
        print(masked_text)

        if mask_positions:
            masked_words = [word for _, word in mask_positions]
            print(f"\n🔍 Masked words: {masked_words}")

        # Generate infills
        print("\n⏳ Generating infills...")
        try:
            results = infill_text(model, tokenizer, additional_tokens_to_ids,
                                  masked_text, mask_types, args.num_infills)

            print("\n✨ PREDICTIONS:")
            if args.markdown:
                # Extract infilled words from each result
                extracted_results = []
                for result in results:
                    # Remove ANSI codes for extraction
                    clean_result = re.sub(r'\033\[[0-9;]*m', '', result)
                    infilled = extract_infilled_words(text, clean_result, mask_positions)
                    extracted_results.append(infilled)

                masked_words = [word for _, word in mask_positions]
                print(format_markdown_table(masked_words, extracted_results, args.num_infills))
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n--- Option {i} ---")
                    print(result)
        except Exception as e:
            print(f"\n❌ Error generating infills: {e}")

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
            # Re-mask the same text (don't increment idx)
            continue
        elif user_input.isdigit():
            idx = int(user_input) - 1  # Convert to 0-indexed
            if idx < 0:
                idx = 0
            elif idx >= len(texts):
                idx = len(texts) - 1
        else:
            idx += 1

    print(f"\nProcessed texts. Final index: {idx + 1}")


if __name__ == '__main__':
    main()
