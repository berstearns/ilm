# EFCAMDAT ILM Training Pipeline - Complete Overview

**NOW INCLUDES CRITICAL GENERAL/BASELINE MODEL**

---

## EXECUTIVE SUMMARY

✅ **PHASE 1 & 2 COMPLETE**: Data preparation & documentation
⏳ **PHASE 3 IN PROGRESS**: Creating masked examples for general model
⏳ **PHASE 4 READY**: Full training pipeline documented

**Total Models**: 6 (not 5!)
- 5 × Per-CEFR specialized models (A1, A2, B1, B2, C1)
- 1 × General baseline model (all CEFR levels mixed)

---

## WHAT'S NEW: THE GENERAL MODEL

### Why It's Critical
```
Research Question: Does specialization actually help?

Per-CEFR Models (Hypothesis):
  - Better on specialized level
  - Risk: C1 has only 10K samples

General Model (Control/Baseline):
  - Single model for all proficiency
  - Tests: Is specialization necessary?
  - Comparison metric for per-CEFR success
```

### Status
```
✅ Data Extracted: 578K train + 72K valid + 72K test
   Location: data/efcamdat_all/
   Size: 185 MB train + 23 MB valid + 23 MB test

⏳ Masked Examples Being Created (9.2M examples for training)
   Duration: ~20 minutes for training set
   Duration: ~2.5 minutes for validation set
   Status: PROCESSING

⏳ Ready to Train: 60,000 steps (5+ hours on V100)
   Output Directory: experiments/efcamdat_all_ilm/
```

---

## COMPLETE TRAINING PIPELINE

### 6 Models Total

| # | Model Name | Dataset | Samples | Train Docs | Examples | Steps | Duration | Purpose |
|---|------------|---------|---------|-----------|----------|-------|----------|---------|
| 1 | efcamdat_C1_ilm | C1 only | 10K | 8K | 128K | 10K | 2-3h | Specialized (advanced) |
| 2 | efcamdat_B2_ilm | B2 only | 40K | 32K | 512K | 20K | 4-5h | Specialized (upper-mid) |
| 3 | efcamdat_B1_ilm | B1 only | 116K | 93K | 1.5M | 30K | 6-8h | Specialized (mid) |
| 4 | efcamdat_A2_ilm | A2 only | 215K | 172K | 2.7M | 40K | 8-10h | Specialized (lower-mid) |
| 5 | efcamdat_A1_ilm | A1 only | 341K | 272K | 4.3M | 50K | 10-12h | Specialized (beginner) |
| 6 | efcamdat_all_ilm | ALL (mixed) | 723K | 578K | 9.2M | 60K | 14-16h | **Baseline (all levels)** |

**Total GPU Time**: ~50-52 hours sequential (can parallelize some)

---

## DATA PREPARATION STATUS

### ✅ COMPLETED

**General Model Data (ALL EFCAMDAT)**:
```
Raw CSV: 723,282 samples
  ├── Downloaded: ✅ Yes
  ├── Loaded: ✅ Yes
  ├── Validated: ✅ Yes

Output Files:
  ├── data/efcamdat_all/train.txt (185.4 MB, 578,625 docs)
  ├── data/efcamdat_all/valid.txt (23.1 MB, 72,328 docs)
  ├── data/efcamdat_all/test.txt (23.3 MB, 72,329 docs)
  └── Total: 231.8 MB raw text

Distribution (Preserved):
  ├── A1: 341,155 (47.2%)
  ├── A2: 215,344 (29.8%)
  ├── B1: 116,539 (16.1%)
  ├── B2: 40,238 (5.6%)
  └── C1: 10,006 (1.4%)

L1 Distribution (Preserved):
  ├── Portuguese: 313,538 (43.3%)
  ├── Mandarin: 129,588 (17.9%)
  ├── Spanish: 64,763 (9.0%)
  ├── Russian: 49,321 (6.8%)
  ├── German: 41,422 (5.7%)
  ├── Italian: 35,428 (4.9%)
  ├── French: 32,519 (4.5%)
  ├── Arabic: 29,308 (4.1%)
  ├── Japanese: 17,086 (2.4%)
  └── Turkish: 10,309 (1.4%)
```

**Test Sample Data**:
```
✅ 100 samples extracted (balanced CEFR distribution)
✅ Masked examples created (1,248 training + 160 validation)
✅ Pipeline validated end-to-end
```

### ⏳ IN PROGRESS

**General Model Masked Examples**:
```
Status: Creating (Background Process 06cbc8 & ef0094)
Training Set:
  ├── Documents to process: 578,625
  ├── Examples per doc: 16
  ├── Total examples: 9,258,000
  ├── Expected output size: ~2.3 GB
  ├── Expected duration: ~20 minutes
  └── Status: RUNNING

Validation Set:
  ├── Documents to process: 72,328
  ├── Examples per doc: 16
  ├── Total examples: 1,157,248
  ├── Expected output size: ~290 MB
  ├── Expected duration: ~2.5 minutes
  └── Status: RUNNING

Expected Completion: ~22 minutes from start time
```

---

## DOCUMENTATION FILES CREATED

### Core Documentation (3 files)

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `efcamdat_training_run.md` | 28 KB | Complete execution guide with all commands | ✅ Updated |
| `strategy_training_groups.md` | 16 KB | Strategic approach (now with 6 models) | ✅ Updated |
| `GENERAL_MODEL_APPROACH.md` | 12 KB | Complete general model strategy | ✅ NEW |

### Reference Documents (4 files)

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `scripts/csv_to_txt_efcamdat.py` | 12 KB | CSV→TXT converter utility | ✅ Tested |
| `IMPLEMENTATION_SUMMARY.md` | Updated | Overall progress tracking | ✅ Updated |
| `PIPELINE_OVERVIEW_WITH_GENERAL_MODEL.md` | This file | Complete pipeline overview | ✅ Current |
| `DETAILED_COMPARISON_TABLE.md` | (pending) | Per-CEFR vs General comparison | 📋 Planned |

**Total Documentation**: ~80 KB, ~3,500 lines, fully executable

---

## EXECUTION WORKFLOW

### Phase 1: Data Preparation ✅ DONE
```
1. Create directories for all 6 models
   ├── data/efcamdat_{all,C1,A2,B1,B2,C1}/
   ├── data/char_masks/efcamdat_{all,C1,A2,B1,B2,C1}/
   └── experiments/efcamdat_{all,C1,A2,B1,B2,C1}_ilm/

2. Extract sample data (100 samples)
   ├── ✅ COMPLETED
   └── Validated pipeline end-to-end

3. Extract general model data (ALL 723K samples)
   ├── ✅ COMPLETED (185 MB)
   └── CEFR distribution preserved

4. Ready to extract per-CEFR datasets
   └── Commands documented in efcamdat_training_run.md Section 2.3-2.7
```

### Phase 2: Create Masked Examples ⏳ IN PROGRESS
```
General Model Examples:
  ├── Training: ⏳ Creating (9.2M examples)
  ├── Validation: ⏳ Creating (1.2M examples)
  └── Expected completion: ~20 minutes

Ready for Per-CEFR Examples:
  ├── Commands documented: ✅ Yes
  ├── Can execute anytime: ✅ Yes
  └── See efcamdat_training_run.md Section 3
```

### Phase 3: Train Models (Recommended Order)
```
Day 1:
  C1 Model:   ⏳ Ready (10K steps, 2-3 hours)
  + Setup time for GPU/environment

Day 2:
  B2 Model:   ⏳ Ready (20K steps, 4-5 hours)
  B1 Model:   ⏳ Ready (30K steps, 6-8 hours)

Day 3:
  A2 Model:   ⏳ Ready (40K steps, 8-10 hours)

Day 4:
  A1 Model:   ⏳ Ready (50K steps, 10-12 hours)

Day 4-5 (Parallel):
  GENERAL Model: ⏳ Ready (60K steps, 14-16 hours)
                   Start while A1 trains if on different GPU

Total: ~4-5 days sequential (can optimize with 2 GPUs)
```

### Phase 4: Evaluation & Comparison ✅ DOCUMENTED
```
Test Each Model:
  ├── C1 model on C1 test set
  ├── B2 model on B2 test set
  ├── B1 model on B1 test set
  ├── A2 model on A2 test set
  ├── A1 model on A1 test set
  └── GENERAL model on all test sets

Cross-Level Evaluation:
  ├── C1 model on A1 test (should be high perplexity)
  ├── A1 model on C1 test (should be high perplexity)
  └── GENERAL model across all levels

Analysis & Comparison:
  ├── Per-CEFR performance vs General model
  ├── Specialization gain (%)
  ├── Cost-benefit analysis
  └── Recommendations

See efcamdat_training_run.md Section 5-6
```

---

## RESEARCH HYPOTHESES

### H1: Specialization Matters ✔️ (Most Likely)
```
Expected Result:
  Per-CEFR models outperform general model on their level
  Example: C1 model PPL ≈ 25 vs General model PPL ≈ 80 on C1 data

Implication:
  Specialization provides 3-8× improvement
  Worth training separate models
  Per-CEFR strategy is superior
```

### H2: General Model is Sufficient ✔️ (Possible)
```
Expected Result:
  General model performance within 10% of best per-CEFR
  Single model simplicity outweighs specialization benefit

Implication:
  One model for all use cases
  Simpler deployment, easier maintenance
  Transfer learning not necessary
```

### H3: Transfer Learning Optimal ✔️ (Advanced)
```
Expected Result:
  Start with general model → fine-tune per-CEFR
  Transfer learning gives best-of-both results

Implication:
  Pre-training valuable (general model)
  Few-shot fine-tuning improves specialization
  Hybrid approach optimal
```

**All three can be true simultaneously** - general model provides baseline for testing them all

---

## KEY METRICS TO TRACK

### Primary Metrics
```
1. Test Set Perplexity (all 6 models on test sets)
   └─ Lower is better

2. Specialization Gain (%)
   └─ (General_PPL / Specialized_PPL - 1) × 100%
   └─ Target: > 25% gain indicates specialization helps

3. Cross-Level Performance Gap
   └─ Model X on Level X vs Model X on Other Levels
   └─ Higher gap indicates specialization
```

### Secondary Metrics
```
1. Training Convergence
   └─ Loss curves, validation improvement over time

2. Transfer Learning Gain
   └─ Performance: General → Fine-tuned vs Direct Training
   └─ Indicates value of pre-training

3. Computational Efficiency
   └─ GPU hours per unit improvement
   └─ Storage required vs performance gain
```

### Deployment Metrics
```
1. Model Size
   └─ 1 general (500MB) vs 5 per-CEFR (2.5GB total)

2. Inference Speed
   └─ Single model vs ensemble performance

3. Cold Start Performance
   └─ How does general model perform on new learner (unknown level)?
```

---

## QUICK START COMMANDS

### Execute Everything In Order (Copy-Paste Ready)

```bash
# 1. Extract all CEFR levels (if not already done)
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_C1 --cefr_level C1 --seed 0

# 2. Create masked examples (general model) - CURRENTLY RUNNING
~/.pyenv/versions/3.9.25/bin/python training/ilm/create_ilm_examples.py \
  train data/char_masks/efcamdat_all --seed 0 --data_name custom --data_dir data/efcamdat_all

# 3. Train general model (14-16 hours)
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_all_ilm training/ilm/train/ data/char_masks/efcamdat_all \
  --seed 0 --train_examples_tag train --eval_examples_tag valid \
  --eval_max_num_examples 500 --model_name gpt2 --train_batch_size 8 --train_num_epochs 1

# 4. Monitor training
tensorboard --logdir experiments --port 6006

# 5. Evaluate general model
~/.pyenv/versions/3.9.25/bin/python training/ilm/train_ilm.py \
  experiments/efcamdat_all_ilm training/ilm/train/ data/char_masks/efcamdat_all \
  --seed 0 --eval_examples_tag test --eval_max_num_examples 1000 --model_name gpt2 --eval_only
```

See `efcamdat_training_run.md` for complete step-by-step guide.

---

## STORAGE REQUIREMENTS

```
Data Phase:
  Raw CSV: 500 MB (original)
  Raw TXT (all levels): ~500 MB

  → Subtotal: ~1 GB

Example Creation Phase:
  Per-CEFR masked examples: ~3.5 GB
  General model masked examples: ~2.6 GB

  → Subtotal: ~6 GB

Training Phase:
  Model checkpoints (6 models): ~3 GB
  TensorBoard logs: ~1 GB

  → Subtotal: ~4 GB

TOTAL REQUIRED: ~11 GB
RECOMMENDED BUFFER: 15-20 GB available space
```

---

## SUCCESS CRITERIA

✅ **Complete if all 6 models**:
- [ ] Trained without CUDA errors
- [ ] Loss decreases during training
- [ ] Validation perplexity reduces
- [ ] Checkpoints saved to experiments/
- [ ] TensorBoard logs created
- [ ] Test set evaluation completed
- [ ] Comparison analysis shows clear winner (per-CEFR vs General)

⭐ **Exceptional if additionally**:
- [ ] Transfer learning tested and documented
- [ ] Cross-level performance gap quantified
- [ ] Per-level performance gain calculated
- [ ] Recommendations for practitioners provided

---

## NEXT IMMEDIATE ACTIONS

1. ⏳ Wait for general model masked examples to complete (~20 min)
2. ✅ Once complete, verify `data/char_masks/efcamdat_all/{train,valid}.pkl` exist
3. 🚀 Begin training general model (longest running)
4. ⏭️ While general trains, extract per-CEFR level data
5. ⏭️ Create masked examples for per-CEFR levels
6. 📊 Evaluate all 6 models when complete

---

## WHY THIS MATTERS

The general model is **not an afterthought** - it's the **control group** in our experiment:

```
Research Question: Is specialization worth it?

Without General Model (Before):
  "We trained 5 per-CEFR models and they work!"
  - Doesn't answer if specialization actually helps
  - No comparison baseline
  - Could use single general model instead

With General Model (Now):
  "We compared 5 per-CEFR models vs 1 general model"
  - Clear evidence if specialization helps (% improvement)
  - Data-driven decision for practitioners
  - Cost-benefit analysis possible
  - Transfer learning hypothesis testable
```

**This changes the research from descriptive to experimental.**

---

**Status**: Ready to execute full 6-model training pipeline
**Files**: All documented with executable commands
**Data**: Ready in data/efcamdat_all/ (general) + ready to extract per-CEFR
**Timeline**: 5-7 days to complete all 6 models + evaluation
