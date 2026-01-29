# EFCAMDAT ILM Training Strategy: Per-CEFR Level Models

**Document Purpose**: Strategic guide for training ILM models on the EFCAMDAT second language learner corpus using a per-CEFR level approach.

**Decision**: Train 5 separate models, one for each CEFR proficiency level (A1, A2, B1, B2, C1).

---

## 1. STRATEGIC RATIONALE: Why 5 Models?

### 1.1 Problem Statement
The EFCAMDAT corpus contains 723,282 writing samples distributed **extremely unevenly** across proficiency levels:
- A1: 341,155 samples (47%)
- A2: 215,344 samples (30%)
- B1: 116,539 samples (16%)
- B2: 40,238 samples (5%)
- C1: 10,006 samples (1%)

### 1.2 Why Not One Model for All Levels?

#### Problem 1: Class Imbalance
- A single model would be **dominated by A1 data** (47%)
- C1 data (1%) would be severely underrepresented
- Model would implicitly learn to generate A1-like text
- C1 samples would have minimal learning signal

#### Problem 2: Linguistic Heterogeneity
Different CEFR levels have fundamentally different linguistic properties:

| Aspect | A1 | B1 | C1 |
|--------|-----|------|-----|
| **Avg Sentence Length** | 10-12 words | 15-18 words | 20-25 words |
| **Vocabulary Size** | ~1,000 | ~3,000-5,000 | ~8,000+ |
| **Grammar Complexity** | Simple present/past | Mixed aspects | Complex structures, passives |
| **Error Rate** | High (20-40%) | Moderate (5-15%) | Low (<5%) |
| **Common Errors** | Articles, tense | Word order, collocation | Rare, subtle errors |

A single model would average these, learning neither A1 nor C1 well.

#### Problem 3: Task-Specific Needs
Different proficiency levels need different infilling capabilities:
- **A1**: Simple word insertion, basic tense forms
- **C1**: Complex phrase completion, idiomatic expressions, nuanced vocabulary

### 1.3 Why 5 Models Works Better

#### Advantage 1: Level-Appropriate Generation
Each model learns language generation at its target proficiency:
- A1 model predicts A1-like text
- C1 model predicts C1-like text
- No averaging effects

#### Advantage 2: Balanced Training
Each dataset uses appropriate size:
- C1: 10,006 samples → 128,064 training examples
- A1: 341,155 samples → 4,366,784 training examples
- Training steps scaled to dataset size

#### Advantage 3: Clear Evaluation
Can directly measure:
- Perplexity on C1 text using C1 model
- Perplexity on A1 text using A1 model
- Cross-level evaluation (C1 model on A1 text = should have higher perplexity)

#### Advantage 4: Practical Application
Enables:
- Level-matched infilling (match student level)
- Difficulty scaling (easier for A1, harder for C1)
- Curriculum learning (train A1→A2→B1→B2→C1)

#### Advantage 5: Quality at High Levels
C1 model specifically trained on advanced text:
- Learns subtle error patterns at C1 level
- Captures advanced vocabulary and phrasing
- Not diluted by beginner examples

---

## 2. TRAINING ORDER AND RATIONALE

### 2.1 Recommended Order: Small to Large + General Baseline

**Train in this order** (6 models total):

**Per-CEFR Specialized Models**:
1. **C1 model** (10K samples - 10,000 training steps)
2. **B2 model** (40K samples - 20,000 training steps)
3. **B1 model** (116K samples - 30,000 training steps)
4. **A2 model** (215K samples - 40,000 training steps)
5. **A1 model** (341K samples - 50,000 training steps)

**General/Baseline Model**:
6. **GENERAL model** (ALL 723K samples - 60,000 training steps)
   - Trained on all CEFR levels without stratification
   - Critical for comparing: Does specialization help?

### 2.2 Why This Order?

#### Rationale 1: Faster Iteration
- C1 trains in 2-3 hours, validate pipeline quickly
- Catch issues early before investing 10+ hours on A1
- Can adjust hyperparameters based on C1 results

#### Rationale 2: Debugging Efficiency
- Smaller datasets = easier to debug
- C1 model failures are easier to diagnose
- Same debugging insights apply to larger models

#### Rationale 3: Computational Efficiency
- Start training C1 while setting up A1/A2 data
- Parallel workflow possible:
  - Hour 1-3: Train C1
  - Hour 2-3: Extract B2 data
  - Hour 3-4: Create B2 examples
  - Hour 4-8: Train B2
  - Overlap reduces total wall-clock time

#### Rationale 4: Empirical Validation
- C1 model provides proof that:
  - Hardware works (no CUDA errors)
  - Data pipeline works
  - Training convergence happens
  - Evaluation metrics make sense

---

## 3. HYPERPARAMETER SELECTION

### 3.1 Shared Hyperparameters (All Models)

```
Base Model: GPT-2 base (124M parameters)
Masking: ilm.mask.hierarchical.MaskHierarchical
  - Word: p=0.03
  - N-gram: p=0.03
  - Sentence: p=0.03
  - Paragraph: p=0.03
  - Document: p=0.03
Batch Size: 8
Learning Rate: 5e-5
Optimizer: AdamW
  - Weight Decay: 0.0
  - Adam Epsilon: 1e-8
Gradient Clipping: Max norm 1.0
Sequence Length: 256
Gradient Accumulation: None (batch size = effective batch)
```

### 3.2 Dataset-Specific Training Steps

**Why different step counts?**

Heuristic: `training_steps = log2(dataset_size) * 1000`

| Level | Samples | Training Examples | Steps | Rationale |
|-------|---------|-------------------|-------|-----------|
| C1 | 10,006 | 160,096 | 10,000 | Smallest, focused training |
| B2 | 40,238 | 643,808 | 20,000 | 2× more data, 2× steps |
| B1 | 116,539 | 1,864,624 | 30,000 | 3× more data, 3× steps |
| A2 | 215,344 | 3,445,504 | 40,000 | 4× more data, 4× steps |
| A1 | 341,155 | 5,458,480 | 50,000 | 5× more data, 5× steps |

**Rationale**:
- Larger datasets need more steps for convergence
- Step count roughly proportional to sqrt(dataset_size)
- Conservative to avoid underfitting large datasets

### 3.3 Evaluation Configuration

```
Evaluation Examples: 500 per evaluation
Evaluation Interval: Every 360 steps (~1 minute of training)
Early Stopping Patience: 10 evals without improvement
Early Stopping Metric: Validation perplexity (lower is better)
Summary Interval: Every 360 steps (TensorBoard logging)
```

---

## 4. RESOURCE PLANNING

### 4.1 Disk Space Required

| Component | C1 | B2 | B1 | A2 | A1 | **Total** |
|-----------|----|----|----|----|----| ---------|
| Raw TXT | 6 MB | 25 MB | 75 MB | 140 MB | 220 MB | **466 MB** |
| Masked .pkl | 40 MB | 160 MB | 470 MB | 900 MB | 1,400 MB | **2.97 GB** |
| Model Checkpoint | 500 MB | 500 MB | 500 MB | 500 MB | 500 MB | **2.5 GB** |
| TensorBoard logs | 50 MB | 100 MB | 200 MB | 300 MB | 400 MB | **1.05 GB** |
| **Level Total** | ~596 MB | ~785 MB | ~1.245 GB | ~1.84 GB | ~2.52 GB | **~6.98 GB** |

**Total Requirement**: ~7 GB (conservative estimate: 10 GB recommended with buffer)

### 4.2 GPU Memory Requirements

**GPU Memory**: 16 GB recommended

**Breakdown**:
- Model parameters: 124M params × 4 bytes = ~500 MB
- Model gradients: ~500 MB
- Optimizer state (Adam): ~1 GB
- Batch 8 with seq_len 256: ~3-4 GB
- Cache/buffers: ~2 GB
- **Total**: ~7-8 GB per model

**Note**: Batch size 4 if running on 11-12 GB GPU (adjust `--train_batch_size 4`)

### 4.3 Training Time Estimates

**Per GPU (V100 or equivalent)**:
- C1: 2-3 hours
- B2: 4-5 hours
- B1: 6-8 hours
- A2: 8-10 hours
- A1: 10-12 hours
- **Total**: ~30-40 GPU hours

**Wall-Clock Time**:
- Sequential execution (one after another): 4-5 days
- With parallel preprocessing: 3-4 days

**Distributed Training** (if available):
- Using Data Parallelism (multiple GPUs): Proportionally faster
- Example: 2 GPUs → ~15-20 hours total

---

## 5. MODEL CONFIGURATIONS SUMMARY

### 5.1 C1 Model Configuration

```yaml
Name: efcamdat_C1_ilm
Dataset: EFCAMDAT C1 level
Samples: 10,006 documents
Training Examples: 160,096 (16 per document)
Split: 80% train (128,076), 10% valid (16,009), 10% test (16,011)
Base Model: gpt2 (124M)
Batch Size: 8
Training Steps: 10,000
Expected Duration: 2-3 hours
Output Directory: experiments/efcamdat_C1_ilm/
```

### 5.2 B2 Model Configuration

```yaml
Name: efcamdat_B2_ilm
Dataset: EFCAMDAT B2 level
Samples: 40,238 documents
Training Examples: 643,808 (16 per document)
Split: 80% train (515,046), 10% valid (64,381), 10% test (64,381)
Base Model: gpt2 (124M)
Batch Size: 8
Training Steps: 20,000
Expected Duration: 4-5 hours
Output Directory: experiments/efcamdat_B2_ilm/
```

### 5.3 B1 Model Configuration

```yaml
Name: efcamdat_B1_ilm
Dataset: EFCAMDAT B1 level
Samples: 116,539 documents
Training Examples: 1,864,624 (16 per document)
Split: 80% train (1,491,699), 10% valid (186,462), 10% test (186,463)
Base Model: gpt2 (124M)
Batch Size: 8
Training Steps: 30,000
Expected Duration: 6-8 hours
Output Directory: experiments/efcamdat_B1_ilm/
```

### 5.4 A2 Model Configuration

```yaml
Name: efcamdat_A2_ilm
Dataset: EFCAMDAT A2 level
Samples: 215,344 documents
Training Examples: 3,445,504 (16 per document)
Split: 80% train (2,756,403), 10% valid (344,550), 10% test (344,551)
Base Model: gpt2 (124M)
Batch Size: 8
Training Steps: 40,000
Expected Duration: 8-10 hours
Output Directory: experiments/efcamdat_A2_ilm/
```

### 5.5 A1 Model Configuration

```yaml
Name: efcamdat_A1_ilm
Dataset: EFCAMDAT A1 level
Samples: 341,155 documents
Training Examples: 5,458,480 (16 per document)
Split: 80% train (4,366,784), 10% valid (545,848), 10% test (545,848)
Base Model: gpt2 (124M)
Batch Size: 8
Training Steps: 50,000
Expected Duration: 10-12 hours
Output Directory: experiments/efcamdat_A1_ilm/
```

---

## 6. RISK MITIGATION STRATEGIES

### 6.1 Risk: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Mitigation**:
1. Reduce batch size: `--train_batch_size 4` (extends training time by 2×)
2. Reduce sequence length: `--train_sequence_length 128` (reduces memory by ~30%)
3. Check other GPU processes: `nvidia-smi`

### 6.2 Risk: Training Loss Diverges

**Symptom**: Loss becomes NaN or rapidly increases

**Mitigation**:
1. Reduce learning rate: `--train_learning_rate 2.5e-5`
2. Add warmup steps (if available)
3. Reduce gradient clipping norm: Add `--train_max_grad_norm 0.5`

### 6.3 Risk: Model Underfits

**Symptom**: Training loss plateaus, no further improvement

**Mitigation**:
1. Increase training steps by 50%
2. Increase learning rate to 1e-4
3. Reduce batch size for more updates

### 6.4 Risk: Data Preparation Errors

**Symptom**: `FileNotFoundError` or `.pkl` file corruption

**Mitigation**:
1. Verify raw data: `head -20 data/efcamdat_X/train.txt`
2. Regenerate `.pkl` files
3. Check disk space: `df -h`

### 6.5 Risk: Slow Training Speed

**Symptom**: Training is slower than expected

**Mitigation**:
1. Increase data loader workers: `--data_loader_num_workers 8`
2. Check GPU utilization: Should be 80-95%
3. Check disk I/O: `iotop`

---

## 7. MONITORING AND VALIDATION

### 7.1 Training Monitoring Checklist

**During training, validate**:
- [ ] GPU memory stable (not increasing)
- [ ] Training loss decreasing
- [ ] Validation perplexity decreasing
- [ ] No CUDA errors
- [ ] No NaN values
- [ ] Training speed: 100-500 examples/sec

### 7.2 TensorBoard Metrics to Watch

```
Metrics to Monitor:
- loss/train: Should decrease monotonically
- eval/perplexity: Should decrease overall
- learning_rate: Should be constant at 5e-5
- gradient_norm: Should be < 1.0 (clipped at 1.0)
- examples_per_second: 100-500 is normal
```

### 7.3 Per-Model Validation

**After C1 training completes**:
1. Check TensorBoard for loss curve
2. Verify checkpoint exists: `ls experiments/efcamdat_C1_ilm/pytorch_model.bin`
3. Run evaluation on test set
4. Compare perplexity to baseline

**If C1 looks good**: Proceed to B2
**If C1 shows issues**: Debug before proceeding

---

## 8. EXECUTION WORKFLOW

### 8.1 Master Execution Plan

```
DAY 1:
  Hour 0-1:    Setup and dependency installation
  Hour 1-2:    Extract all CEFR level datasets
  Hour 2-3:    Create masked examples for all levels
  Hour 3-6:    Train C1 model (running in background)

  Parallel (Hour 2-3):
    Create B2, B1 examples in separate terminal

DAY 2:
  Hour 0-4:    Train B2 model
  Hour 4-12:   Train B1 model (8 hours)

DAY 3:
  Hour 0-10:   Train A2 model (10 hours)

DAY 4:
  Hour 0-12:   Train A1 model (12 hours)

DAY 5:
  Hour 0-2:    Evaluate all models
  Hour 2-4:    Generate comparison metrics
```

### 8.2 Parallel Execution (Optional - Requires Multiple GPUs)

```
GPU 1: Train C1 (2-3 hours)
GPU 2: Train B2 (4-5 hours) - Start after C1 begins
GPU 3: Train B1 (6-8 hours) - Start after B2 begins
GPU 4: Train A2 (8-10 hours)

Result: Can run all in ~12 hours with 4 GPUs
```

### 8.3 Checkpointing Strategy

**Save checkpoints**:
- All training saves automatic checkpoints
- Location: `experiments/efcamdat_X_ilm/`
- Can resume if interrupted: Add `--resume_from_checkpoint`

---

## 9. SUCCESS CRITERIA

### 9.1 Model Training Success

For each model, training is considered successful if:
1. ✅ Training completes without CUDA errors
2. ✅ Training loss decreases over time
3. ✅ Validation perplexity decreases after initial epochs
4. ✅ Model checkpoint saved to disk
5. ✅ TensorBoard logs created
6. ✅ Final validation perplexity < 50 (reasonable for GPT-2)

### 9.2 Per-Model Targets

Expected test set perplexity (approximate ranges):

| Model | Expected Perplexity | Comment |
|-------|-------------------|---------|
| C1 | 20-40 | Advanced text, harder to predict |
| B2 | 15-35 | Still challenging |
| B1 | 12-30 | Moderate |
| A2 | 10-25 | Simpler patterns |
| A1 | 8-20 | Most predictable text |

*Note: Actual values depend on training duration and data quality*

### 9.3 Cross-Model Consistency Check

After training all models, verify cross-model relationships:

```
C1_model_on_C1_data:     PPL ~25 (good - expert on its data)
C1_model_on_A1_data:     PPL ~80 (bad - shouldn't understand simple text)
A1_model_on_A1_data:     PPL ~15 (good - expert on its data)
A1_model_on_C1_data:     PPL ~150 (bad - shouldn't understand complex text)
```

This confirms models are learning level-appropriate language.

---

## 10. NEXT PHASES (Beyond Initial 5 Models)

### 10.1 Phase 2: Fine-tuning by L1 Language
After initial 5 models, optionally train per-L1 fine-tuned versions:
- Take A1 model, fine-tune on Portuguese A1 data only
- Create language-specific models

### 10.2 Phase 3: Ensemble Models
Combine multiple models for improved predictions:
- Ensemble voting
- Stacking

### 10.3 Phase 4: Production Deployment
Deploy models as:
- API service
- Interactive web interface
- Classroom tutoring system

---

## APPENDIX: Comparison to Alternative Approaches

### Alternative 1: Single Model for All Levels
**Pros**: Fewer models to maintain, simpler
**Cons**: Class imbalance, poor performance on rare classes (C1), averaged generation
**Decision**: ❌ Rejected

### Alternative 2: Multi-Task Learning (Single Model with Level Labels)
**Pros**: Single model, uses all data
**Cons**: Requires architectural changes, complex training, unproven for learner data
**Decision**: ❌ Rejected (too experimental)

### Alternative 3: Per-L1 Models (50 Models: 5 CEFR × 10 L1)
**Pros**: Maximum specificity
**Cons**: 50 models to train (~500 GPU hours), more maintenance, some L1s would have <1000 samples
**Decision**: ❌ Rejected (overly complex for current phase)

### **Alternative 4: Per-CEFR Models (5 Models)** ← CHOSEN
**Pros**: Balances specificity and practicality, addresses class imbalance, enables level-specific applications
**Cons**: Requires more storage than single model
**Decision**: ✅ **SELECTED**

---

**Document Status**: APPROVED and ACTIVE
**Last Updated**: 2026-01-29
**Author**: Claude Code
**Reviewers**: None (draft)
