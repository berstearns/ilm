#!/usr/bin/env python3
"""
Convert EFCAMDAT CSV to ILM-compatible TXT format.

EFCAMDAT corpus contains English writing samples from learners.
This script extracts data and converts it to the format expected by create_ilm_examples.py:
- Documents separated by triple newlines (\n\n\n)
- Split into train.txt, valid.txt, test.txt files
- Optional stratified sampling by CEFR level and L1 language
"""

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import pandas as pd
import numpy as np


def load_efcamdat_csv(csv_path: str) -> pd.DataFrame:
    """
    Load EFCAMDAT CSV file.

    Expected columns: writing_id, l1, cefr_level, text

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with columns: writing_id, l1, cefr_level, text
    """
    print(f"Loading EFCAMDAT CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['writing_id', 'l1', 'cefr_level', 'text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove rows with missing text
    df = df.dropna(subset=['text'])

    # Clean text: strip whitespace, remove empty strings
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 0]

    print(f"Loaded {len(df)} samples")
    return df


def stratified_sample_by_cefr(
    df: pd.DataFrame,
    cefr_level: str,
    sample_size: int,
    seed: int = 0
) -> pd.DataFrame:
    """
    Extract stratified sample for a specific CEFR level.

    Args:
        df: Input DataFrame with all data
        cefr_level: CEFR level to extract (A1, A2, B1, B2, C1)
        sample_size: Total number of samples to extract
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled data for specified CEFR level
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter to specific CEFR level
    df_cefr = df[df['cefr_level'] == cefr_level].copy()

    if len(df_cefr) == 0:
        raise ValueError(f"No samples found for CEFR level: {cefr_level}")

    print(f"CEFR Level {cefr_level}: {len(df_cefr)} total samples available")

    # If sample_size >= available samples, use all
    if sample_size >= len(df_cefr):
        print(f"Using all {len(df_cefr)} available samples")
        return df_cefr

    # Stratified sampling by L1 language within this CEFR level
    # This ensures we get a representative sample of L1 languages
    samples_per_l1 = defaultdict(int)
    l1_counts = df_cefr['l1'].value_counts().to_dict()

    # Distribute sample_size proportionally among L1 languages
    total_l1 = len(df_cefr)
    samples_by_l1 = {}
    remaining = sample_size

    # Sort L1s by count (largest first) for deterministic allocation
    sorted_l1s = sorted(l1_counts.items(), key=lambda x: x[1], reverse=True)

    for i, (l1, count) in enumerate(sorted_l1s):
        if i == len(sorted_l1s) - 1:
            # Last L1 gets remaining samples
            samples_by_l1[l1] = remaining
        else:
            # Proportional allocation
            n_samples = max(1, int(sample_size * count / total_l1))
            samples_by_l1[l1] = min(n_samples, count, remaining)
            remaining -= samples_by_l1[l1]

    # Sample from each L1 group
    samples = []
    for l1, n_samples in samples_by_l1.items():
        l1_data = df_cefr[df_cefr['l1'] == l1]
        sampled = l1_data.sample(n=min(n_samples, len(l1_data)), random_state=seed)
        samples.append(sampled)
        print(f"  L1 {l1}: {len(sampled)} samples")

    result = pd.concat(samples, ignore_index=True)

    # Shuffle to mix L1s
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Total sample for {cefr_level}: {len(result)} samples")
    return result


def create_train_valid_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/valid/test sets.

    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        valid_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed

    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    # Validate ratios sum to 1.0
    total = train_ratio + valid_ratio + test_ratio
    if not (0.99 < total < 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = df[:train_end]
    valid_df = df[train_end:valid_end]
    test_df = df[valid_end:]

    print(f"Train: {len(train_df)} samples ({100*len(train_df)/n:.1f}%)")
    print(f"Valid: {len(valid_df)} samples ({100*len(valid_df)/n:.1f}%)")
    print(f"Test:  {len(test_df)} samples ({100*len(test_df)/n:.1f}%)")

    return train_df, valid_df, test_df


def write_ilm_format(
    df: pd.DataFrame,
    output_dir: str,
    split_name: str,
    text_column: str = 'text'
) -> None:
    """
    Write DataFrame to ILM-compatible TXT format.

    Format: Documents separated by triple newlines (\n\n\n)

    Args:
        df: DataFrame with text data
        output_dir: Directory to write output file
        split_name: Name of split (train, valid, test)
        text_column: Name of column containing text
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{split_name}.txt')

    # Extract texts and join with triple newlines
    texts = df[text_column].tolist()
    combined_text = '\n\n\n'.join(texts)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"Wrote {output_path}")

    # Print file statistics
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    n_docs = len(texts)
    avg_len = np.mean([len(t) for t in texts])

    print(f"  - {n_docs} documents")
    print(f"  - {file_size_mb:.1f} MB")
    print(f"  - {avg_len:.0f} chars/doc average")


def print_data_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics about the dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print("\nCEFR Level Distribution:")
    cefr_counts = df['cefr_level'].value_counts().sort_index()
    for cefr, count in cefr_counts.items():
        pct = 100 * count / len(df)
        print(f"  {cefr}: {count:>7} ({pct:>5.1f}%)")

    print("\nL1 Language Distribution:")
    l1_counts = df['l1'].value_counts()
    for l1, count in l1_counts.items():
        pct = 100 * count / len(df)
        print(f"  {l1}: {count:>7} ({pct:>5.1f}%)")

    print("\nText Length Statistics:")
    text_lengths = df['text'].str.len()
    print(f"  Min: {text_lengths.min()} chars")
    print(f"  Max: {text_lengths.max()} chars")
    print(f"  Mean: {text_lengths.mean():.0f} chars")
    print(f"  Median: {text_lengths.median():.0f} chars")
    print(f"  Std Dev: {text_lengths.std():.0f} chars")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert EFCAMDAT CSV to ILM-compatible TXT format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Extract 100 balanced samples from all CEFR levels
python csv_to_txt_efcamdat.py \\
    --csv_path /path/to/norm-EFCAMDAT-ALL-CONCAT.csv \\
    --output_dir data/efcamdat_samples \\
    --sample_size 100 \\
    --seed 0

# Extract all A1 level samples
python csv_to_txt_efcamdat.py \\
    --csv_path /path/to/norm-EFCAMDAT-ALL-CONCAT.csv \\
    --output_dir data/efcamdat_A1 \\
    --cefr_level A1 \\
    --seed 0

# Extract 50 samples per CEFR level (250 total)
python csv_to_txt_efcamdat.py \\
    --csv_path /path/to/norm-EFCAMDAT-ALL-CONCAT.csv \\
    --output_dir data/efcamdat_mixed \\
    --sample_size 250 \\
    --seed 0
        """
    )

    parser.add_argument(
        '--csv_path',
        required=True,
        help='Path to EFCAMDAT CSV file'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for train.txt, valid.txt, test.txt'
    )
    parser.add_argument(
        '--cefr_level',
        default=None,
        help='Extract only specific CEFR level (A1, A2, B1, B2, C1). If None, uses all levels.'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Sample size. If None, uses all available data.'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Proportion of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.1,
        help='Proportion of data for validation (default: 0.1)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Proportion of data for testing (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Load CSV
    df = load_efcamdat_csv(args.csv_path)

    # Apply CEFR level filter if specified
    if args.cefr_level:
        df = stratified_sample_by_cefr(
            df,
            cefr_level=args.cefr_level,
            sample_size=args.sample_size or len(df),
            seed=args.seed
        )
    elif args.sample_size:
        # Sample proportionally from all CEFR levels
        cefr_levels = df['cefr_level'].unique()

        samples = []
        for level in sorted(cefr_levels):
            level_count = len(df[df['cefr_level'] == level])
            level_sample_size = max(1, int(args.sample_size * level_count / len(df)))

            sampled = stratified_sample_by_cefr(
                df,
                cefr_level=level,
                sample_size=level_sample_size,
                seed=args.seed + ord(level[0])  # Vary seed by level
            )
            samples.append(sampled)

        df = pd.concat(samples, ignore_index=True)
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Print statistics
    print_data_statistics(df)

    # Create train/valid/test split
    print("Creating train/valid/test split...")
    train_df, valid_df, test_df = create_train_valid_test_split(
        df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Write files in ILM format
    print("\nWriting ILM-compatible TXT files...")
    write_ilm_format(train_df, args.output_dir, 'train')
    write_ilm_format(valid_df, args.output_dir, 'valid')
    write_ilm_format(test_df, args.output_dir, 'test')

    print(f"\nSuccess! Output files written to: {args.output_dir}")
    print(f"  - {args.output_dir}/train.txt")
    print(f"  - {args.output_dir}/valid.txt")
    print(f"  - {args.output_dir}/test.txt")


if __name__ == '__main__':
    main()
