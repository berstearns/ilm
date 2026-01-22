# -*- coding: utf-8 -*-
"""ILM (Infilling Language Model) Inference Script

Based on: https://colab.research.google.com/drive/1So95M0hHefyNm_eELglCna_ZayoDX6KV
Requires: Python 3.9.x, transformers==2.7.0, ilm package
"""

import os
import pickle

import torch
from transformers import GPT2LMHeadModel

import ilm.tokenize_util
from ilm.infer import infill_with_ilm


def main():
    # Configuration - use project's models folder
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'sto_ilm')

    # Check if model exists
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}. Run setup.sh first to download the model."
        )

    # Prepare tokenizer
    print("Loading tokenizer...")
    tokenizer = ilm.tokenize_util.Tokenizer.GPT2

    additional_ids_to_tokens_path = os.path.join(MODEL_DIR, 'additional_ids_to_tokens.pkl')
    with open(additional_ids_to_tokens_path, 'rb') as f:
        additional_ids_to_tokens = pickle.load(f)
    additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}

    try:
        ilm.tokenize_util.update_tokenizer(additional_ids_to_tokens, tokenizer)
    except ValueError:
        print('Tokenizer already updated')

    print(f"Additional tokens: {additional_tokens_to_ids}")

    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    model.to(device)

    # Create context with blanks
    context = """
Math Class
Chris was bad at _. _ _ _ He ended up passing the test.
""".strip()

    print(f"\nOriginal context:\n{context}\n")

    # Tokenize context
    context_ids = ilm.tokenize_util.encode(context, tokenizer)

    # Replace blanks with appropriate infill tokens from left to right
    _blank_id = ilm.tokenize_util.encode(' _', tokenizer)[0]

    # First blank: word-level infill
    context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_word|>']
    # Next three blanks: sentence-level infills
    context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
    context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']
    context_ids[context_ids.index(_blank_id)] = additional_tokens_to_ids['<|infill_sentence|>']

    print(f"Context with infill tokens:\n{ilm.tokenize_util.decode(context_ids, tokenizer)}\n")

    # Generate infills
    print("Generating infills...")
    generated = infill_with_ilm(
        model,
        additional_tokens_to_ids,
        context_ids,
        num_infills=2
    )

    # Print results
    print("\n" + "=" * 80)
    print("GENERATED INFILLS:")
    print("=" * 80)
    for i, g in enumerate(generated, 1):
        print(f"\n--- Generation {i} ---")
        print(ilm.tokenize_util.decode(g, tokenizer))
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
