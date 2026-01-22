# -*- coding: utf-8 -*-
"""Interactive ILM Inference Script

Iterates over texts from a CSV file, randomly masks tokens, and shows predictions.
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
    Finds where ' _' was replaced and highlights those portions in green.
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
    """Load the ILM model and tokenizer."""
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
    """Get positions of words in text (start, end, word)."""
    positions = []
    for match in re.finditer(r'\b\w+\b', text):
        positions.append((match.start(), match.end(), match.group()))
    return positions


def mask_random_words(text, n_masks=3, mask_type='word'):
    """
    Randomly mask n words in the text.
    Returns: (masked_text, mask_positions, mask_types)
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
    """Infill the masked text."""
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
    Extract the infilled words from a result by comparing with original.
    Returns list of infilled words in order of mask positions.
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
    """Format predictions as a markdown table."""
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
    """Truncate text to max_chars, trying to end at a sentence boundary."""
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
    """Get default model directory based on project structure."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, 'models', 'sto_ilm')


def main():
    parser = argparse.ArgumentParser(description='Interactive ILM Inference')
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
