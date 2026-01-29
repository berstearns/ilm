# ILM Training and Evaluation Examples

This document contains all commands for training new ILM models and evaluating them on various tasks.
All examples use Python 3.9 from ~/.pyenv/versions/3.9.25/bin/python

---

## SECTION 1: ENVIRONMENT SETUP AND INSTALLATION

### 1.1 Verify Python Version
Check that you're using the correct Python interpreter (3.9.25):
```bash
~/.pyenv/versions/3.9.25/bin/python --version
# Output: Python 3.9.25
```

### 1.2 Create Project Directory Structure
Set up the working directories for data, models, and results:
```bash
mkdir -p /home/b/p/research-sketches/ilms/data
mkdir -p /home/b/p/research-sketches/ilms/models
mkdir -p /home/b/p/research-sketches/ilms/results
mkdir -p /home/b/p/research-sketches/ilms/data/char_masks
cd /home/b/p/research-sketches/ilms
```

### 1.3 Install Dependencies with Python 3.9
Install all required packages using the specific Python version:
```bash
~/.pyenv/versions/3.9.25/bin/python -m pip install --upgrade pip setuptools wheel
~/.pyenv/versions/3.9.25/bin/python -m pip install -r requirements.txt
```

### 1.4 Download NLTK Data
Required for tokenization and sentence segmentation:
```bash
~/.pyenv/versions/3.9.25/bin/python -c "import nltk; nltk.download('punkt')"
```

### 1.5 Install ILM Package in Development Mode
Install the training/ilm package locally:
```bash
cd training/ilm
~/.pyenv/versions/3.9.25/bin/python -m pip install -e .
cd /home/b/p/research-sketches/ilms
```

---

## SECTION 2: DATA PREPARATION (Dataset Download & Processing)

### 2.1 Download Arxiv CS Abstracts Dataset
Download abstracts from computer science papers:
```bash
cd training/ilm/data
bash get_arxiv_cs_abstracts.sh
cd /home/b/p/research-sketches/ilms
```

### 2.2 Download ROC Stories Dataset
Download story data for story infilling:
```bash
cd training/ilm/data
bash get_roc_stories.sh
cd /home/b/p/research-sketches/ilms
```

### 2.3 Download Song Lyrics Dataset
Download song lyrics stanzas:
```bash
cd training/ilm/data
bash get_lyrics_stanzas.sh
cd /home/b/p/research-sketches/ilms
```

### 2.4 Prepare Custom Dataset
To use your own text data:
1. Create three files: train.txt, valid.txt, test.txt
2. Each file should contain documents separated by triple newlines (\n\n\n)
3. Place files in: training/ilm/data/custom_dataset/
```bash
mkdir -p training/ilm/data/custom_dataset
# Example: creating train.txt with two documents
cat > training/ilm/data/custom_dataset/train.txt << 'EOF'
First document goes here.
It can have multiple sentences.


Second document goes here.
Also multiple sentences allowed.
EOF
```

---

## SECTION 3: CREATE ILM TRAINING EXAMPLES (Data Preprocessing)

This stage transforms raw text into training examples with masked spans.

### 3.1 Create ILM Examples from Arxiv CS Abstracts - Training Set
Generate training examples with word/sentence/paragraph/document masking:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --data_name arxiv_cs_abstracts \
  --data_split train
```

Parameters explained:
- `train`: Task type (train/valid/test)
- `training/ilm/data/char_masks/arxiv_cs_abstracts`: Output directory for masked examples
- `--seed 0`: Random seed for reproducibility
- `--data_name arxiv_cs_abstracts`: Which dataset to use
- `--data_split train`: Which split of the dataset (train/valid/test)

Output files created:
- `training/ilm/data/char_masks/arxiv_cs_abstracts/train.pkl`
- Contains pairs of (documents, character-level masks)

### 3.2 Create ILM Examples from Arxiv CS Abstracts - Validation Set
Generate validation examples:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --data_name arxiv_cs_abstracts \
  --data_split valid
```

Output: `training/ilm/data/char_masks/arxiv_cs_abstracts/valid.pkl`

### 3.3 Create ILM Examples from Arxiv CS Abstracts - Test Set
Generate test examples:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  test \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --data_name arxiv_cs_abstracts \
  --data_split test
```

Output: `training/ilm/data/char_masks/arxiv_cs_abstracts/test.pkl`

### 3.4 Create ILM Examples from ROC Stories - Training Set
Generate examples from story dataset:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  training/ilm/data/char_masks/roc_stories \
  --seed 0 \
  --data_name roc_stories \
  --data_split train
```

Output: `training/ilm/data/char_masks/roc_stories/train.pkl`

### 3.5 Create ILM Examples from ROC Stories - Validation Set
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  training/ilm/data/char_masks/roc_stories \
  --seed 0 \
  --data_name roc_stories \
  --data_split valid
```

Output: `training/ilm/data/char_masks/roc_stories/valid.pkl`

### 3.6 Create ILM Examples from Custom Dataset
Generate examples from your custom dataset:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  training/ilm/data/char_masks/custom_dataset \
  --seed 0 \
  --data_name custom_dataset \
  --data_split train
```

### 3.7 Preview Created Examples
Inspect the examples before training to verify quality:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/preview_ilm_examples.py \
  training/ilm/data/char_masks/arxiv_cs_abstracts/train.pkl \
  --num_examples 5
```

This shows 5 examples of how documents were masked for training.

---

## SECTION 4: TRAIN A NEW ILM MODEL (Core Training)

### 4.1 Train ILM Model on Arxiv CS Abstracts
Full training example with default hyperparameters:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --eval_interval_steps 360 \
  --summary_interval_steps 360
```

Parameters explained:
- `experiments/arxiv_ilm_base`: Output directory for model checkpoints
- `training/ilm/train/`: Training script directory
- `training/ilm/data/char_masks/arxiv_cs_abstracts`: Directory with .pkl files
- `--seed 0`: Random seed for reproducibility
- `--train_examples_tag train`: Use train.pkl examples
- `--eval_examples_tag valid`: Use valid.pkl for evaluation
- `--eval_max_num_examples 512`: Max examples per eval (512 is recommended)
- `--gpt2_model_name gpt2`: Use GPT-2 base (options: gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- `--batch_size 8`: Batch size for training
- `--num_train_steps 50000`: Total training steps
- `--eval_interval_steps 360`: Evaluate every 360 steps (~1 minute)
- `--summary_interval_steps 360`: Log summaries every 360 steps

Output files:
- `experiments/arxiv_ilm_base/config.json`: Model configuration
- `experiments/arxiv_ilm_base/pytorch_model.bin`: Trained weights
- `experiments/arxiv_ilm_base/optimizer.pt`: Optimizer state
- `experiments/arxiv_ilm_base/step.pkl`: Current training step
- `experiments/arxiv_ilm_base/tfevents-*`: TensorBoard logs

### 4.2 Train ILM Model on ROC Stories (Smaller Dataset)
Train a model for story infilling with reduced steps:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/stories_ilm_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/roc_stories \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 256 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 20000
```

### 4.3 Train Large ILM Model (GPT-2 Large)
Train with larger model size for better quality (requires more GPU memory):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_large \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2-large \
  --batch_size 4 \
  --num_train_steps 50000
```

Parameters changed:
- `--gpt2_model_name gpt2-large`: Use 774M parameter model
- `--batch_size 4`: Reduced from 8 to 4 (larger model needs more VRAM)

### 4.4 Train with Gradient Accumulation (Higher Effective Batch Size)
Train with gradient accumulation for larger effective batch size:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_accumulated \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --gradient_accumulation_steps 4
```

Parameters:
- `--gradient_accumulation_steps 4`: Accumulate gradients over 4 steps (effective batch = 8*4 = 32)

### 4.5 Train with Custom Learning Rate
Train with lower learning rate for more stable training:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_lowlr \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --learning_rate 1e-5
```

Parameters:
- `--learning_rate 1e-5`: Use 1e-5 instead of default 5e-5

### 4.6 Train with Early Stopping on Custom Metric
Train with patient early stopping on validation loss:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_earlystop \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --eval_patience 10
```

Parameters:
- `--eval_patience 10`: Stop if validation loss doesn't improve for 10 evaluations

### 4.7 Resume Training from Checkpoint
Resume training from an existing checkpoint:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 100000 \
  --resume_from_checkpoint
```

Parameters:
- `--resume_from_checkpoint`: Auto-loads from experiments/arxiv_ilm_base/pytorch_model.bin

### 4.8 Train with Weights & Biases Experiment Tracking
Track experiments with W&B for monitoring:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --wandb
```

Note: Requires `pip install wandb` and `wandb login`

### 4.9 Train with Different Task Types
Train with NAIVE task (full document as target instead of just infill):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_naive_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --task_type naive
```

Parameters:
- `--task_type naive`: Available: ilm (default), naive, lm, reverse_lm, no_context_ilm

### 4.10 Train with Context Loss
Train by also computing loss on context tokens (not just infill):
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_context \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000 \
  --train_context_loss
```

Parameters:
- `--train_context_loss`: Also compute loss on context tokens (default: False)

---

## SECTION 5: EVALUATE TRAINED MODEL

### 5.1 Evaluate on Arxiv Test Set
Evaluate the trained model on test set:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_base \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag test \
  --eval_examples_tag test \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --num_train_steps 0 \
  --eval_only
```

This runs evaluation only without training steps.

---

## SECTION 6: INFERENCE AND EVALUATION WITH TRAINED MODELS

### 6.1 Run Evaluation on Pretrained Story ILM Model
Evaluate the pretrained story infilling model on CSV data:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-celva-1742.csv \
  --models ilm:models/sto_ilm mlm:bert-base-uncased mlm:roberta-base t5:t5-small mlm:answerdotai/ModernBERT-base \
  --n-masks 1 \
  --samples-per-text 10 \
  --max-chars 500 \
  --limit 5 \
  --seed 42 \
  --print-every 10 \
  --masking human-tokens \
  --subtoken-granularity both \
  -o results/results-human-10.json
```

Parameters explained:
- `-i`: Input CSV file path
- `--models`: Model specifications in format name:model_path
  - `ilm:models/sto_ilm`: Pretrained story ILM
  - `mlm:bert-base-uncased`: BERT base uncased
  - `mlm:roberta-base`: RoBERTa base
  - `t5:t5-small`: T5 small
  - `mlm:answerdotai/ModernBERT-base`: ModernBERT from HuggingFace
- `--n-masks 1`: Number of masks per span
- `--samples-per-text 10`: 10 samples per text
- `--max-chars 500`: Max 500 characters per text
- `--limit 5`: Limit to first 5 texts
- `--seed 42`: Random seed
- `--print-every 10`: Print progress every 10 examples
- `--masking human-tokens`: Mask strategy (human-tokens or random)
- `--subtoken-granularity both`: Granularity (both, character, subword)
- `-o`: Output JSON file

### 6.2 Run Evaluation on Newly Trained Model
Evaluate a newly trained model:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-celva-1742.csv \
  --models ilm:experiments/arxiv_ilm_base mlm:bert-base-uncased \
  --n-masks 1 \
  --samples-per-text 10 \
  --max-chars 500 \
  --limit 5 \
  --seed 42 \
  --masking human-tokens \
  -o results/results-arxiv-trained.json
```

Just replace the model path with your trained model directory.

### 6.3 Evaluate Multiple Models Simultaneously
Compare multiple trained models:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-celva-1742.csv \
  --models \
    ilm:models/sto_ilm \
    ilm:experiments/arxiv_ilm_base \
    ilm:experiments/stories_ilm_base \
    mlm:bert-base-uncased \
  --n-masks 1 \
  --samples-per-text 5 \
  --limit 10 \
  --seed 42 \
  -o results/comparison-all-models.json
```

### 6.4 Run Interactive Infilling with Trained Model
Use the trained model for interactive infilling:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_interactive.py \
  --model_path experiments/arxiv_ilm_base \
  --device cuda
```

This starts an interactive session where you can type text and the model fills in masked spans.

### 6.5 Run Interactive Infilling with Pretrained Model
Interactive session with pretrained story model:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_interactive.py \
  --model_path models/sto_ilm \
  --device cuda
```

---

## SECTION 7: REPRODUCE ACL 2020 PAPER RESULTS

### 7.1 Generate All Commands for Paper Reproduction
Generate all commands for paper models:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/acl20_repro_train.py \
  --output_dir acl20_models
```

This generates all download + training commands for:
- Abstracts, Stories, Lyrics datasets
- LM, LMRev, LMAll, ILM task types
- Word, N-gram, Sentence, Paragraph, Document masking

### 7.2 Reproduce Paper Evaluation Metrics
Test that perplexity matches paper:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/acl20_repro_eval.py \
  --model_dir acl20_models/arxiv_abstracts_ilm_word
```

---

## SECTION 8: ADVANCED TRAINING SCENARIOS

### 8.1 Train with Different Mask Granularities
Train on document-level masking only:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  training/ilm/data/char_masks/arxiv_documents_only \
  --seed 0 \
  --data_name arxiv_cs_abstracts \
  --data_split train \
  --mask_type document
```

Then train:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/arxiv_ilm_docs_only \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_documents_only \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 512 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 50000
```

### 8.2 Ensemble Different Models
Combine predictions from multiple models:
```bash
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \
  -i /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-celva-1742.csv \
  --models \
    ilm:models/sto_ilm \
    ilm:experiments/arxiv_ilm_base \
    ilm:experiments/stories_ilm_base \
    ilm:experiments/arxiv_ilm_large \
  --n-masks 1 \
  --samples-per-text 5 \
  --limit 100 \
  --ensemble_method average \
  -o results/ensemble-results.json
```

### 8.3 Fine-tune Pretrained Model on Custom Data
Fine-tune story model on custom dataset:
```bash
# First create examples from custom data
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train \
  training/ilm/data/char_masks/custom_dataset \
  --seed 0 \
  --data_name custom_dataset \
  --data_split train

~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  valid \
  training/ilm/data/char_masks/custom_dataset \
  --seed 0 \
  --data_name custom_dataset \
  --data_split valid

# Then fine-tune pretrained model
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/story_finetuned_custom \
  training/ilm/train/ \
  training/ilm/data/char_masks/custom_dataset \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 256 \
  --gpt2_model_name gpt2 \
  --batch_size 8 \
  --num_train_steps 10000 \
  --learning_rate 2e-5 \
  --pretrained_model_path models/sto_ilm
```

Parameters:
- `--pretrained_model_path models/sto_ilm`: Load pretrained weights instead of random init

---

## SECTION 9: TROUBLESHOOTING AND MONITORING

### 9.1 Monitor Training with TensorBoard
View training metrics in real-time:
```bash
# In a separate terminal:
tensorboard --logdir experiments/arxiv_ilm_base --port 6006
# Then visit http://localhost:6006 in browser
```

### 9.2 Check GPU Memory Usage During Training
Monitor GPU during training:
```bash
# In a separate terminal, run nvidia-smi continuously:
watch -n 1 nvidia-smi
```

### 9.3 Debug by Testing on Small Data
Quick test with just 10 examples:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/test_small \
  training/ilm/train/ \
  training/ilm/data/char_masks/arxiv_cs_abstracts \
  --seed 0 \
  --train_examples_tag train \
  --eval_examples_tag valid \
  --eval_max_num_examples 10 \
  --gpt2_model_name gpt2 \
  --batch_size 2 \
  --num_train_steps 100 \
  --eval_interval_steps 10
```

### 9.4 Check Syntax of ILM Examples
Preview examples to ensure they're correct:
```bash
~/.pyenv/versions/3.9.25/bin/python training/ilm/preview_ilm_examples.py \
  training/ilm/data/char_masks/arxiv_cs_abstracts/train.pkl \
  --num_examples 10
```

### 9.5 Verify Model Architecture
Print loaded model architecture:
```bash
~/.pyenv/versions/3.9.25/bin/python -c "
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
print(model)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## QUICK REFERENCE: ESSENTIAL COMMANDS SUMMARY

### Full Training Pipeline
```bash
# 1. Setup
cd training/ilm && ~/.pyenv/versions/3.9.25/bin/python -m pip install -e . && cd ../..

# 2. Download data
cd training/ilm/data && bash get_arxiv_cs_abstracts.sh && cd ../../..

# 3. Create examples
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py train training/ilm/data/char_masks/arxiv_cs_abstracts --data_name arxiv_cs_abstracts --data_split train --seed 0
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py valid training/ilm/data/char_masks/arxiv_cs_abstracts --data_name arxiv_cs_abstracts --data_split valid --seed 0

# 4. Train model
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py experiments/arxiv_ilm_base training/ilm/train/ training/ilm/data/char_masks/arxiv_cs_abstracts --seed 0 --train_examples_tag train --eval_examples_tag valid --eval_max_num_examples 512 --gpt2_model_name gpt2 --batch_size 8 --num_train_steps 50000

# 5. Evaluate
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py -i /path/to/data.csv --models ilm:experiments/arxiv_ilm_base --n-masks 1 --limit 10 -o results/eval.json
```

---

## COMMON CONFIGURATIONS

### Minimal Training (Test Only)
- Dataset: Arxiv CS Abstracts
- Model: GPT-2 base
- Batch size: 8
- Steps: 50K (1-2 hours on V100)
- **Best for**: Quick iteration

### Production Training
- Dataset: Arxiv CS Abstracts
- Model: GPT-2 large
- Batch size: 4 (with grad accumulation 4)
- Steps: 200K (8-12 hours on V100)
- Eval patience: 10
- **Best for**: High quality models

### Story Infilling
- Dataset: ROC Stories
- Model: GPT-2 base
- Batch size: 8
- Steps: 20K
- Learning rate: 5e-5
- **Best for**: Story completion tasks

---

## FILE ORGANIZATION AFTER FULL PIPELINE

```
/home/b/p/research-sketches/ilms/
├── training/ilm/
│   ├── train_ilm.py                    # Training script
│   ├── create_ilm_examples.py          # Example generation
│   ├── ilm/                            # Core package
│   └── data/
│       ├── arxiv_cs_abstracts.tar.gz   # Downloaded data
│       ├── roc_stories.tar.gz
│       └── char_masks/
│           ├── arxiv_cs_abstracts/
│           │   ├── train.pkl           # Created examples
│           │   └── valid.pkl
│           └── roc_stories/
│               ├── train.pkl
│               └── valid.pkl
├── experiments/
│   ├── arxiv_ilm_base/
│   │   ├── pytorch_model.bin           # Trained model
│   │   ├── config.json
│   │   ├── optimizer.pt
│   │   └── tfevents-*
│   └── stories_ilm_base/
│       └── ...
├── models/
│   └── sto_ilm/                        # Pretrained model
│       ├── pytorch_model.bin
│       └── config.json
├── results/
│   ├── results-human-10.json           # Eval results
│   └── eval.json
└── examples_run.md                     # This file
```

---

## EXAMPLE: COMPLETE TRAINING SESSION TRANSCRIPT

This is a realistic example of all commands run in sequence:

```bash
# Navigate to project
cd /home/b/p/research-sketches/ilms

# Verify Python
~/.pyenv/versions/3.9.25/bin/python --version
# Python 3.9.25

# Install
cd training/ilm
~/.pyenv/versions/3.9.25/bin/python -m pip install -r requirements.txt
~/.pyenv/versions/3.9.25/bin/python -m pip install -e .
cd ../..

# Download data
cd training/ilm/data
bash get_arxiv_cs_abstracts.sh
cd ../../..

# Create examples
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py train training/ilm/data/char_masks/arxiv_cs_abstracts --seed 0 --data_name arxiv_cs_abstracts --data_split train
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py valid training/ilm/data/char_masks/arxiv_cs_abstracts --seed 0 --data_name arxiv_cs_abstracts --data_split valid

# Train model
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py experiments/arxiv_ilm_base training/ilm/train/ training/ilm/data/char_masks/arxiv_cs_abstracts --seed 0 --train_examples_tag train --eval_examples_tag valid --eval_max_num_examples 512 --gpt2_model_name gpt2 --batch_size 8 --num_train_steps 50000

# Watch training (in another terminal)
tensorboard --logdir experiments/arxiv_ilm_base --port 6006

# After training, evaluate
~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py -i my_test_data.csv --models ilm:experiments/arxiv_ilm_base mlm:bert-base-uncased -o results/evaluation.json --limit 100

# Try interactive infilling
~/.pyenv/versions/3.9.25/bin/python inference/ilm_interactive.py --model_path experiments/arxiv_ilm_base --device cuda
```

