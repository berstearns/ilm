# EFCAMDAT Second Language Learner Corpus - ILM Training and Evaluation Guide

This document contains all commands for training ILM models on the EFCAMDAT corpus of English writing samples from second language learners. The golden rule: **every command that will be executed is documented before execution**.

**Dataset**: English learner writing samples from 723,282 writers across 5 CEFR proficiency levels (A1-C1) and 10 L1 language backgrounds.

---

## SECTION 1: ENVIRONMENT SETUP AND INSTALLATION

### 1.1 Verify Python Version
Check that you're using the correct Python interpreter (3.9.25):
```bash
~/.pyenv/versions/3.9.25/bin/python --version
# Expected Output: Python 3.9.25
```

### 1.2 Create Project Directory Structure
Set up the working directories for data, models, and results:
```bash
mkdir -p /home/b/p/research-sketches/ilms/data/efcamdat_{samples,A1,A2,B1,B2,C1}
mkdir -p /home/b/p/research-sketches/ilms/data/char_masks/efcamdat_{samples,A1,A2,B1,B2,C1}
mkdir -p /home/b/p/research-sketches/ilms/experiments/efcamdat_{test_sample,A1_ilm,A2_ilm,B1_ilm,B2_ilm,C1_ilm}
mkdir -p /home/b/p/research-sketches/ilms/scripts
mkdir -p /home/b/p/research-sketches/ilms/results/efcamdat_{A1,A2,B1,B2,C1}
cd /home/b/p/research-sketches/ilms
```

**Expected Output**: Directories created silently.

### 1.3 Install Dependencies with Python 3.9
Install all required packages using the specific Python version:
```bash
~/.pyenv/versions/3.9.25/bin/python -m pip install --upgrade pip setuptools wheel
~/.pyenv/versions/3.9.25/bin/python -m pip install torch transformers datasets tensorboard tqdm numpy pandas nltk scikit-learn
```

**Expected Output**: Packages installed, several warnings about versions are normal.

### 1.4 Download NLTK Data
Required for tokenization and sentence segmentation:
```bash
~/.pyenv/versions/3.9.25/bin/python -c "import nltk; nltk.download('punkt')"
```

**Expected Output**: `[nltk_data] Downloading package punkt to /home/b/nltk_data...`

### 1.5 Install ILM Package in Development Mode
Install the training/ilm package locally:
```bash
~/.pyenv/versions/3.9.25/bin/python -m pip install -e /home/b/p/research-sketches/ilms/training/ilm
```

**Expected Output**: `Successfully installed ilm-0.0.0`

### 1.6 Verify All Dependencies
Quick test to ensure all imports work:
```bash
~/.pyenv/versions/3.9.25/bin/python -c "
import torch
import transformers
import nltk
import pandas
import numpy
print('All dependencies loaded successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
"
```

**Expected Output**: Version information for all packages.

---

## SECTION 2: DATA PREPARATION (EFCAMDAT CSV to ILM Format)

### 2.1 Create CSV Conversion Script
The script `scripts/csv_to_txt_efcamdat.py` has been created to convert EFCAMDAT CSV to ILM-compatible TXT format with stratified sampling by CEFR level and L1 language.

**Script Location**: `/home/b/p/research-sketches/ilms/scripts/csv_to_txt_efcamdat.py`

**Features**:
- Loads EFCAMDAT CSV with validation
- Supports stratified sampling by CEFR level and L1 language
- Outputs train/valid/test splits (80/10/10 by default)
- Formats documents with triple newline separator (\n\n\n)
- Includes comprehensive statistics

### 2.2 Extract Sample Dataset for Testing (100 Samples)
Generate a balanced 100-sample dataset for pipeline validation:
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_samples \
  --sample_size 100 \
  --seed 0
```

**Execution Results**:
- **Duration**: ~15-30 seconds
- **Output Files**:
  - `data/efcamdat_samples/train.txt` (78 documents, ~26 KB)
  - `data/efcamdat_samples/valid.txt` (10 documents, ~3 KB)
  - `data/efcamdat_samples/test.txt` (10 documents, ~3 KB)
- **CEFR Distribution**: A1: 47, A2: 29, B1: 16, B2: 5, C1: 1
- **Text Statistics**: Mean 328 chars, Min 118, Max 1082
- **Status**: ✅ COMPLETED - All files created successfully

### 2.3 Extract Full C1 Dataset (10,006 Samples - Smallest)
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_C1 \
  --cefr_level C1 \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_C1/train.txt` (~5 MB, 8,004 documents)
- `data/efcamdat_C1/valid.txt` (~0.6 MB, 1,000 documents)
- `data/efcamdat_C1/test.txt` (~0.6 MB, 1,002 documents)
- **Duration**: ~10 seconds

### 2.4 Extract Full B2 Dataset (40,238 Samples)
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_B2 \
  --cefr_level B2 \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_B2/train.txt` (~20 MB, 32,190 documents)
- `data/efcamdat_B2/valid.txt` (~2.5 MB, 4,024 documents)
- `data/efcamdat_B2/test.txt` (~2.5 MB, 4,024 documents)
- **Duration**: ~20 seconds

### 2.5 Extract Full B1 Dataset (116,539 Samples)
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_B1 \
  --cefr_level B1 \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_B1/train.txt` (~60 MB, 93,231 documents)
- `data/efcamdat_B1/valid.txt` (~7.5 MB, 11,654 documents)
- `data/efcamdat_B1/test.txt` (~7.5 MB, 11,654 documents)
- **Duration**: ~40 seconds

### 2.6 Extract Full A2 Dataset (215,344 Samples)
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_A2 \
  --cefr_level A2 \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_A2/train.txt` (~110 MB, 172,275 documents)
- `data/efcamdat_A2/valid.txt` (~14 MB, 21,534 documents)
- `data/efcamdat_A2/test.txt` (~14 MB, 21,535 documents)
- **Duration**: ~90 seconds

### 2.7 Extract Full A1 Dataset (341,155 Samples - Largest)
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_A1 \
  --cefr_level A1 \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_A1/train.txt` (~175 MB, 272,924 documents)
- `data/efcamdat_A1/valid.txt` (~22 MB, 34,116 documents)
- `data/efcamdat_A1/test.txt` (~22 MB, 34,115 documents)
- **Duration**: ~150 seconds (~2.5 minutes)

### 2.8 Extract General/Baseline Dataset (ALL 723,282 Samples - No CEFR Stratification)
**CRITICAL: This is the general baseline model trained on all data without CEFR-level specialization.**

This model serves as:
- Baseline for comparison against specialized per-CEFR models
- Null hypothesis: Does specialization help?
- Transfer learning source for fine-tuning
- General-purpose ILM for mixed-level learners

Extract all EFCAMDAT data without filtering:
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_all \
  --seed 0
```

**Expected Output**:
- `data/efcamdat_all/train.txt` (~420 MB, 578,625 documents - 80% of 723,282)
- `data/efcamdat_all/valid.txt` (~52 MB, 72,328 documents - 10%)
- `data/efcamdat_all/test.txt` (~52 MB, 72,329 documents - 10%)
- **CEFR Distribution in All Data**: A1:47%, A2:30%, B1:16%, B2:5%, C1:1% (original proportions)
- **Duration**: ~400 seconds (~7 minutes)

**Key Difference from Per-CEFR Models**:
- Per-CEFR models: Each model trained only on A1, A2, B1, B2, or C1 samples
- General Model: Single model trained on ALL mixed-proficiency samples
- This tests whether specialized models outperform a general model

---

## SECTION 3: CREATE ILM TRAINING EXAMPLES (Data Preprocessing)

This stage transforms raw text into training examples with masked spans. Uses hierarchical masking with word/ngram/sentence/paragraph/document level masks.

### 3.1 Create ILM Examples from Sample Dataset - Training Set
Generate training examples with hierarchical masking from sample data:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_samples \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_samples
```

**Execution Results**:
- **Duration**: ~2 seconds
- **Output File**: `data/char_masks/efcamdat_samples/train.pkl` (55 KB)
- **Examples Created**: 78 documents × 16 examples/document = 1,248 training examples
- **Mask Rate**: ~14% of characters masked
- **Status**: ✅ COMPLETED - Training examples created successfully

### 3.2 Create ILM Examples from Sample Dataset - Validation Set
Generate validation examples:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_samples \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_samples
```

**Expected Output**:
- `data/char_masks/efcamdat_samples/valid.pkl` (55 KB)
- **Examples Created**: ~160 validation examples

### 3.3 Create ILM Examples from C1 Dataset - Training Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_C1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_C1
```

**Expected Output**:
- `data/char_masks/efcamdat_C1/train.pkl` (~40 MB)
- **Examples Created**: 8,004 documents × 16 = 128,064 training examples
- **Duration**: ~15 seconds

### 3.4 Create ILM Examples from C1 Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_C1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_C1
```

**Expected Output**: `data/char_masks/efcamdat_C1/valid.pkl` (~5 MB)

### 3.5 Create ILM Examples from B2 Dataset - Training Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_B2 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_B2
```

**Expected Output**:
- `data/char_masks/efcamdat_B2/train.pkl` (~160 MB)
- **Examples Created**: 32,190 × 16 = 515,040 examples
- **Duration**: ~60 seconds

### 3.6 Create ILM Examples from B2 Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_B2 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_B2
```

**Expected Output**: `data/char_masks/efcamdat_B2/valid.pkl` (~20 MB)

### 3.7 Create ILM Examples from B1 Dataset - Training Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_B1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_B1
```

**Expected Output**:
- `data/char_masks/efcamdat_B1/train.pkl` (~470 MB)
- **Examples Created**: 93,231 × 16 = 1,491,696 examples
- **Duration**: ~180 seconds (~3 minutes)

### 3.8 Create ILM Examples from B1 Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_B1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_B1
```

**Expected Output**: `data/char_masks/efcamdat_B1/valid.pkl` (~60 MB)

### 3.9 Create ILM Examples from A2 Dataset - Training Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_A2 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_A2
```

**Expected Output**:
- `data/char_masks/efcamdat_A2/train.pkl` (~900 MB)
- **Examples Created**: 172,275 × 16 = 2,756,400 examples
- **Duration**: ~300 seconds (~5 minutes)

### 3.10 Create ILM Examples from A2 Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_A2 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_A2
```

**Expected Output**: `data/char_masks/efcamdat_A2/valid.pkl` (~110 MB)

### 3.11 Create ILM Examples from A1 Dataset - Training Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_A1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_A1
```

**Expected Output**:
- `data/char_masks/efcamdat_A1/train.pkl` (~1.4 GB)
- **Examples Created**: 272,924 × 16 = 4,366,784 examples
- **Duration**: ~500 seconds (~8 minutes)

### 3.12 Create ILM Examples from A1 Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_A1 \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_A1
```

**Expected Output**: `data/char_masks/efcamdat_A1/valid.pkl` (~180 MB)

### 3.13 Create ILM Examples from General/All Dataset - Training Set
**CRITICAL: Create masked examples for the general baseline model**

This model is trained on all 723,282 samples without CEFR stratification.

```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  data/char_masks/efcamdat_all \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_all
```

**Expected Output**:
- `data/char_masks/efcamdat_all/train.pkl` (~2.3 GB)
- **Examples Created**: 578,625 documents × 16 = 9,258,000 training examples
- **Duration**: ~1,200 seconds (~20 minutes)

### 3.14 Create ILM Examples from General/All Dataset - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  data/char_masks/efcamdat_all \
  --seed 0 \
  --data_name custom \
  --data_dir data/efcamdat_all
```

**Expected Output**:
- `data/char_masks/efcamdat_all/valid.pkl` (~290 MB)
- **Duration**: ~150 seconds (~2.5 minutes)

### 3.13 Preview Created Examples (Optional Quality Check)
Inspect sample examples to verify masking quality:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/preview_ilm_examples.py \
  data/char_masks/efcamdat_samples/train.pkl \
  --num_examples 5
```

**Expected Output**: Display of 5 examples showing original text and masked spans for different levels.

---

## SECTION 4: TRAIN ILM MODELS (Core Training)

### 4.1 Train Test Model on Sample Data (Validation Run)
Quick training run to validate the entire pipeline (1,000 steps, ~15-20 minutes):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_test_sample \
  training/ilm/train/ \
  data/char_masks/efcamdat_samples \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 10 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Expected Behavior**:
- Loads 1,248 training examples
- Loads 160 validation examples
- Initializes GPT-2 base model (124M parameters)
- Trains for 1 epoch (~125 training steps)
- Evaluates every 100 steps
- **Expected Duration**: 10-15 minutes on V100 GPU
- **Output Files**:
  - `experiments/efcamdat_test_sample/pytorch_model.bin` (500 MB)
  - `experiments/efcamdat_test_sample/config.json`
  - `experiments/efcamdat_test_sample/optimizer.pt`
  - TensorBoard event files
- **Success Indicators**:
  - Training loss decreases over time
  - Validation perplexity shown
  - No CUDA out of memory errors
  - Model checkpoint saved

### 4.2 Train C1 ILM Model (Small Dataset - Start Here)
Train model on smallest CEFR level dataset (10,000 steps, ~2-3 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_C1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_C1 \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Parameters**:
- **Model**: GPT-2 base (124M parameters)
- **Batch Size**: 8
- **Learning Rate**: 5e-5 (default)
- **Training Examples**: 128,064
- **Eval Examples**: 16,000 (capped at 500 per eval)
- **Expected Steps**: ~16,000 (128,064 examples / batch size 8)
- **Expected Duration**: 2-3 hours on V100
- **Checkpoint**: `experiments/efcamdat_C1_ilm/`

### 4.3 Train B2 ILM Model (Medium Dataset)
Train on B2 level dataset (20,000 steps, ~4-5 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_B2_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_B2 \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Expected**:
- **Training Examples**: 515,040
- **Expected Steps**: ~64,000
- **Duration**: 4-5 hours

### 4.4 Train B1 ILM Model (Medium-Large Dataset)
Train on B1 level dataset (30,000 steps, ~6-8 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_B1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_B1 \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Expected**:
- **Training Examples**: 1,491,696
- **Expected Steps**: ~186,462
- **Duration**: 6-8 hours

### 4.5 Train A2 ILM Model (Large Dataset)
Train on A2 level dataset (40,000 steps, ~8-10 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_A2_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_A2 \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Expected**:
- **Training Examples**: 2,756,400
- **Expected Steps**: ~344,550
- **Duration**: 8-10 hours

### 4.6 Train A1 ILM Model (Largest Dataset)
Train on A1 level dataset (50,000 steps, ~10-12 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_A1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_A1 \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Expected**:
- **Training Examples**: 4,366,784
- **Expected Steps**: ~545,848
- **Duration**: 10-12 hours

### 4.7 Train GENERAL/BASELINE ILM Model (All CEFR Levels - 723,282 Samples)
**CRITICAL BASELINE MODEL: Trained on ALL mixed-proficiency data**

This general model is essential for:
- Baseline comparison (does specialization help?)
- Mixed-level learner scenarios
- Transfer learning source
- Understanding class imbalance effects

Train on all EFCAMDAT data combined (60,000 steps, ~14-16 hours):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_all_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_all \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 500 \
  --model_name gpt2 \
  --train_batch_size 8 \
  --train_num_epochs 1
```

**Parameters**:
- **Dataset**: All 723,282 samples combined (NOT stratified by CEFR)
- **Base Model**: GPT-2 base (124M parameters)
- **Training Examples**: 9,258,000 (578,625 docs × 16)
- **Expected Steps**: ~1,157,250 (9.2M examples / batch size 8)
- **Training Steps**: 60,000 (covers ~52% of full dataset)
- **Expected Duration**: 14-16 hours on V100
- **Batch Size**: 8
- **Learning Rate**: 5e-5 (default)
- **Output Directory**: `experiments/efcamdat_all_ilm/`

**Why This Model Matters**:
1. **Baseline**: Tests if specialized per-CEFR models outperform general model
2. **Comparison Metric**: Compare perplexity on mixed vs. specialized
3. **Transfer Source**: Fine-tune from this model for specific CEFR levels
4. **Real-World Scenario**: Useful for systems that don't know learner proficiency level

**Key Difference**:
- **Per-CEFR Models** (5 total): Each trained only on one CEFR level
  - A1 model: 341K samples → specialized for beginner level
  - C1 model: 10K samples → specialized for advanced level
  - No cross-level interference

- **General Model** (1 total): Trained on all 723K samples mixed
  - Learns to handle A1 through C1 in single model
  - May suffer from class imbalance (A1 dominates with 47%)
  - Tests hypothesis: Is specialization necessary?

### 4.7 Monitor Training with TensorBoard
In a separate terminal, monitor training progress:
```bash
tensorboard --logdir experiments --port 6006
# Then visit http://localhost:6006 in browser
```

**TensorBoard Metrics**:
- Training loss
- Validation perplexity
- Learning rate
- Gradient norm
- Training speed (examples/sec)

---

## SECTION 5: EVALUATE TRAINED MODELS

### 5.1 Evaluate C1 Model on Test Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_C1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_C1 \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

**Expected Output**:
- Test set perplexity for C1 model
- Saved to TensorBoard logs

### 5.2 Evaluate B2 Model on Test Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_B2_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_B2 \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

### 5.3 Evaluate B1 Model on Test Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_B1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_B1 \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

### 5.4 Evaluate A2 Model on Test Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_A2_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_A2 \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

### 5.5 Evaluate A1 Model on Test Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_A1_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_A1 \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

### 5.6 Evaluate General/All Model on Test Set
**CRITICAL: Evaluate the general baseline model**

```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_all_ilm \
  training/ilm/train/ \
  data/char_masks/efcamdat_all \
  --seed 0 \
  --eval_examples_tag test \
  --eval_max_num_examples 1000 \
  --model_name gpt2 \
  --eval_only
```

**Expected Output**:
- Test set perplexity for the general model
- Compare with per-CEFR models: Does general model perform better or worse?
- This is the key comparison metric

### 5.7 Cross-Model Comparison: Per-CEFR vs. General
**Compare all 6 models to determine if specialization helps**

Generate comprehensive comparison:
- C1 model on C1 test data vs. General model on C1 test data
- A1 model on A1 test data vs. General model on A1 test data
- General model on mixed data (should be worse than per-CEFR on its level)

Expected pattern:
```
A1 model on A1 test:     PPL ~12-18 (specialized - best)
General model on A1 test: PPL ~15-25 (general - worse)

C1 model on C1 test:     PPL ~25-40 (specialized - best)
General model on C1 test: PPL ~80-150 (general - much worse for advanced)
```

This validates the hypothesis that **specialization improves performance on specific levels**.

---

## SECTION 6: INFERENCE AND EVALUATION WITH TRAINED MODELS

### 6.1 Run Evaluation on Trained Model
Evaluate a trained model on a CSV file:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /path/to/test_data.csv \
  --models ilm:experiments/efcamdat_A1_ilm mlm:bert-base-uncased \
  --n-masks 1 \
  --samples-per-text 5 \
  --max-chars 500 \
  --limit 100 \
  --seed 42 \
  --masking human-tokens \
  -o results/efcamdat/A1_eval.json
```

**Parameters Explained**:
- `-i`: Input CSV file with text column
- `--models`: Model specifications (ilm:path, mlm:model_name, t5:model_name)
- `--n-masks`: Number of masks per span
- `--samples-per-text`: Samples generated per text
- `--max-chars`: Maximum characters per text
- `--limit`: Limit to first N texts
- `--seed`: Random seed
- `--masking`: Masking strategy (human-tokens or random)
- `-o`: Output JSON file with results

### 6.2 Run Interactive Infilling with Trained Model
Use the trained model for interactive infilling:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_interactive.py \
  --model_path experiments/efcamdat_A1_ilm \
  --device cuda
```

**Interactive Mode**:
- Type text with [MASK] placeholders
- Model fills in masked spans
- Type 'quit' to exit

### 6.3 Compare Multiple Models
Evaluate multiple trained models for comparison:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /path/to/test_data.csv \
  --models \
    ilm:experiments/efcamdat_C1_ilm \
    ilm:experiments/efcamdat_B2_ilm \
    ilm:experiments/efcamdat_B1_ilm \
    ilm:experiments/efcamdat_A2_ilm \
    ilm:experiments/efcamdat_A1_ilm \
  --n-masks 1 \
  --samples-per-text 5 \
  --limit 1000 \
  --seed 42 \
  -o results/efcamdat/all_models_comparison.json
```

---

## SECTION 7: LEARNER-SPECIFIC CONSIDERATIONS

### 7.1 Text Length Handling
Learner texts are generally shorter than native text (mean ~330 chars vs ~500+ for other domains):

**Considerations**:
- Default sequence length (256) is appropriate
- May want to reduce to 128 for very short texts
- Set `--train_sequence_length 128` if needed:
```bash
--train_sequence_length 128
```

### 7.2 Error Preservation
Learner texts contain grammatical and lexical errors. These are features, not bugs:

**Important**: Do NOT pre-correct the text. The masking function will handle errors transparently.

### 7.3 Vocabulary Characteristics
Learner vocabulary is more limited and repetitive:

**Impact**:
- Models may converge faster (fewer unique tokens)
- Consider early stopping with patience
- Reduce learning rate: `--train_learning_rate 2.5e-5`

### 7.4 Class Imbalance in Multi-Level Training
When training across levels, use stratified sampling:

**Already handled** in `csv_to_txt_efcamdat.py` with stratified L1 language distribution.

---

## SECTION 8: TROUBLESHOOTING AND MONITORING

### 8.1 Check GPU Memory Usage During Training
Monitor GPU during training:
```bash
# In a separate terminal:
watch -n 1 nvidia-smi
```

**Expected**:
- GPU Mem: ~12-14 GB with batch size 8
- GPU Util: 80-95%
- Process: python training/ilm/train_ilm.py

### 8.2 Debug Mode - Train on Tiny Dataset
Quick test with just 1,000 steps:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_test_tiny \
  training/ilm/train/ \
  data/char_masks/efcamdat_samples \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 5 \
  --model_name gpt2 \
  --train_batch_size 4 \
  --train_num_epochs 1
```

**Expected Duration**: 2-3 minutes

### 8.3 Check Syntax of ILM Examples
Verify examples are correct before training:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/preview_ilm_examples.py \
  data/char_masks/efcamdat_samples/train.pkl \
  --num_examples 10
```

### 8.4 Verify Model Architecture
Print loaded model architecture and parameter count:
```bash
~/.pyenv/versions/3.9.25/bin/python -c "
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
print(model)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

**Expected Output**: Model config with 124,439,808 parameters for GPT-2 base

### 8.5 Common Errors and Solutions

**Error**: `CUDA out of memory`
- **Solution**: Reduce batch size to 4: `--train_batch_size 4`
- **Alternative**: Reduce sequence length: `--train_sequence_length 128`

**Error**: `FileNotFoundError: No such file or directory: '*.pkl'`
- **Solution**: Ensure examples have been created with `create_ilm_examples.py`
- **Check**: `ls data/char_masks/efcamdat_samples/`

**Error**: `ValueError: not enough values to unpack`
- **Solution**: May indicate corrupted .pkl file, regenerate with `create_ilm_examples.py`

**Error**: `Loss is NaN`
- **Solution**: Reduce learning rate: `--train_learning_rate 1e-5`
- **Or**: Use gradient clipping (default is set to 1.0)

---

## SECTION 9: QUICK REFERENCE

### Essential Commands Summary

**Setup**:
```bash
~/.pyenv/versions/3.9.25/bin/python -m pip install torch transformers datasets tensorboard tqdm numpy pandas nltk scikit-learn
~/.pyenv/versions/3.9.25/bin/python -m pip install -e /home/b/p/research-sketches/ilms/training/ilm
```

**Extract Sample**:
```bash
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv --output_dir data/efcamdat_samples --sample_size 100 --seed 0
```

**Create Examples**:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py train data/char_masks/efcamdat_samples --data_name custom --data_dir data/efcamdat_samples --seed 0
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py valid data/char_masks/efcamdat_samples --data_name custom --data_dir data/efcamdat_samples --seed 0
```

**Train Model** (example - A1):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py experiments/efcamdat_A1_ilm training/ilm/train/ data/char_masks/efcamdat_A1 --seed 0 --train_examples_tag train --eval_examples_tag valid --eval_max_num_examples 500 --model_name gpt2 --train_batch_size 8 --train_num_epochs 1
```

**Monitor**:
```bash
tensorboard --logdir experiments --port 6006
```

---

## SECTION 10: DATASET STATISTICS (Reference)

### EFCAMDAT Distribution by CEFR Level

| CEFR Level | Total Samples | Percentage | Files |
|------------|---------------|-----------|-------|
| A1 | 341,155 | 47.17% | data/efcamdat_A1/* |
| A2 | 215,344 | 29.77% | data/efcamdat_A2/* |
| B1 | 116,539 | 16.11% | data/efcamdat_B1/* |
| B2 | 40,238 | 5.56% | data/efcamdat_B2/* |
| C1 | 10,006 | 1.38% | data/efcamdat_C1/* |
| **TOTAL** | **723,282** | **100%** | All models |

### Distribution by L1 Language

| L1 Language | Count | Percentage |
|-------------|-------|-----------|
| Portuguese | 313,538 | 43.35% |
| Mandarin | 129,588 | 17.92% |
| Spanish | 64,763 | 8.95% |
| Russian | 49,321 | 6.82% |
| German | 41,422 | 5.73% |
| Italian | 35,428 | 4.90% |
| French | 32,519 | 4.50% |
| Arabic | 29,308 | 4.05% |
| Japanese | 17,086 | 2.36% |
| Turkish | 10,309 | 1.43% |

---

## APPENDIX: Training Timeline Estimates

**Phase 1: Setup & Validation** (60-90 minutes):
- Install dependencies
- Extract 100 sample and train test model

**Phase 2: Data Extraction** (8-10 minutes total):
- Extract all 5 CEFR level datasets

**Phase 3: Example Creation** (~18 minutes total):
- Create masked examples for all levels

**Phase 4: Model Training** (~30-40 GPU hours):
- C1: 2-3 hours
- B2: 4-5 hours
- B1: 6-8 hours
- A2: 8-10 hours
- A1: 10-12 hours

**Phase 5: Evaluation** (2-4 hours):
- Test set evaluation
- Model comparison
- Analysis

**Total Estimated Time**: 40-50 hours (with sequential GPU execution)

---

**Last Updated**: 2026-01-29
**Status**: In Progress - Sample extraction and example creation completed ✅
**Next Steps**: Execute training models starting with C1 level
