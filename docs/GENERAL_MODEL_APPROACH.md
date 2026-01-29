# EFCAMDAT General/Baseline Model - Strategy Document

**Critical Addition**: A 6th model trained on ALL 723,282 samples without CEFR stratification

---

## 1. WHY THE GENERAL MODEL IS ESSENTIAL

### The Research Question
**Does training separate models for each CEFR level actually outperform a single general model trained on all mixed-proficiency data?**

### Current Hypothesis
Per-CEFR models should outperform because:
- Each model specializes on one proficiency level
- No class imbalance (equal treatment of A1-C1)
- Level-specific error patterns learned
- Level-appropriate language generation

### But We Need to Test This
A general baseline model provides:
1. **Null hypothesis**: What if a general model works just as well?
2. **Comparison metric**: How much does specialization help (% improvement)?
3. **Transfer learning source**: Fine-tune for specific levels
4. **Real-world baseline**: What happens when proficiency level is unknown?

---

## 2. DATASET COMPOSITION COMPARISON

### Per-CEFR Models
```
C1 Model:   10,006 samples (1.4%) → Trained only on C1
B2 Model:   40,238 samples (5.6%) → Trained only on B2
B1 Model:  116,539 samples (16%) → Trained only on B1
A2 Model:  215,344 samples (30%) → Trained only on A2
A1 Model:  341,155 samples (47%) → Trained only on A1
```

**Characteristics**:
- ✅ Each model is expert on its level
- ✅ No class imbalance within model
- ❌ C1 model has only 10K samples (small dataset risk)
- ❌ A1 model has bias toward beginner patterns

### General Model
```
General Model: 723,282 samples (100%) → Trained on ALL mixed
  - 47% A1 samples
  - 30% A2 samples
  - 16% B1 samples
  - 5.6% B2 samples
  - 1.4% C1 samples
```

**Characteristics**:
- ✅ Large dataset (723K samples)
- ✅ More robust learning signal (9.2M examples)
- ✅ May discover cross-level patterns
- ❌ Dominated by beginner examples (47% A1)
- ❌ Advanced patterns (C1) diluted by noise

---

## 3. EXPECTED OUTCOMES & HYPOTHESES

### Hypothesis A: Specialization Matters (Most Likely)
```
Metric: Test Set Perplexity

C1 Model Performance:
  - On C1 test data:     PPL ≈ 25-40 (GOOD - specialized)
  - On A1 test data:     PPL ≈ 150-300 (BAD - confused by simple text)

A1 Model Performance:
  - On A1 test data:     PPL ≈ 12-20 (GOOD - specialized)
  - On C1 test data:     PPL ≈ 200-400 (BAD - can't handle complexity)

General Model Performance:
  - On A1 test data:     PPL ≈ 15-25 (mediocre - average of all)
  - On C1 test data:     PPL ≈ 80-150 (mediocre - compromised)
  - Overall PPL across all:  ~50-80 (averaged)

Result: Per-CEFR models WIN
  - C1 model is 3-6x better at C1
  - A1 model is 1.5-2x better at A1
  - Specialization provides clear benefit
```

### Hypothesis B: General Model is Sufficient
```
Metric: Test Set Perplexity

General Model achieves:
  - PPL ≈ 15-25 across all levels (surprisingly good)
  - Better at A1/A2 due to class weight
  - Acceptable at C1 (doesn't have to learn from 10K tiny dataset)

Result: General Model TIES or WINS
  - Single model simplicity valuable
  - Per-CEFR models no better
  - Transfer learning not necessary
```

### Hypothesis C: Transfer Learning is Key
```
Hypothesis: Start with general model, fine-tune per-CEFR

Fine-tuned Model Performance:
  - General → A1: PPL ≈ 11-18 (even better than A1-only)
  - General → C1: PPL ≈ 20-35 (better than C1-only)

Result: Transfer learning WINS
  - General model as pre-training
  - Few-shot fine-tuning with CEFR data
  - Best of both worlds
```

---

## 4. EXPERIMENTAL DESIGN

### Models to Train (6 Total)

```
Specialized Models (5):
  1. efcamdat_C1_ilm   - Trained on C1 only (10K samples)
  2. efcamdat_B2_ilm   - Trained on B2 only (40K samples)
  3. efcamdat_B1_ilm   - Trained on B1 only (116K samples)
  4. efcamdat_A2_ilm   - Trained on A2 only (215K samples)
  5. efcamdat_A1_ilm   - Trained on A1 only (341K samples)

Baseline Model (1):
  6. efcamdat_all_ilm  - Trained on ALL (723K samples)
```

### Evaluation Protocol

**Test 1: Within-Level Performance**
```
C1 model on C1 test (?):  PPL_c1_c1
A1 model on A1 test (?):  PPL_a1_a1
General model on C1 (?):  PPL_gen_c1
General model on A1 (?):  PPL_gen_a1

Specialization Gain = (PPL_gen_c1 / PPL_c1_c1 - 1) * 100%
  e.g., if PPL_gen=100, PPL_c1=25, gain = (100/25 - 1) = 300% improvement
```

**Test 2: Cross-Level Performance**
```
A1 model on C1 test (?):  PPL_a1_c1 (should be high - wrong level)
C1 model on A1 test (?):  PPL_c1_a1 (should be high - wrong level)
General model on C1 (?):  PPL_gen_c1 (should be mid - understands some complexity)
General model on A1 (?):  PPL_gen_a1 (should be low - understands basics)
```

**Test 3: Mixed-Level Performance**
```
All models on combined test set (all levels mixed)
General model should perform best (trained on mixed distribution)
Per-CEFR models should average worse (each misses non-target levels)
```

**Test 4: Transfer Learning**
```
Start with general model weights
Fine-tune on A1 data (10K steps)
Evaluate on A1 test set
Compare to A1-only model

If transfer > A1-only: Transfer learning is valuable
If A1-only > transfer: Direct training is better
```

---

## 5. COMPUTATIONAL COSTS

### Training Time
```
C1 Model:      10,000 steps ×  0.05 sec/step = 8 minutes (extrapolated)
B2 Model:      20,000 steps ×  0.10 sec/step = 33 minutes
B1 Model:      30,000 steps ×  0.15 sec/step = 75 minutes
A2 Model:      40,000 steps ×  0.20 sec/step = 2.6 hours
A1 Model:      50,000 steps ×  0.25 sec/step = 3.5 hours

General Model: 60,000 steps ×  0.30 sec/step = 5.0 hours
             (larger dataset = slower per-step)

Total: ~14-16 hours GPU time (all sequential)
```

### Storage
```
Per-CEFR Models:
  C1: 40 MB + 500 MB checkpoint = 540 MB
  B2: 160 MB + 500 MB checkpoint = 660 MB
  B1: 470 MB + 500 MB checkpoint = 970 MB
  A2: 900 MB + 500 MB checkpoint = 1.4 GB
  A1: 1.4 GB + 500 MB checkpoint = 1.9 GB
  Subtotal: ~5.5 GB

General Model:
  Training data: 2.3 GB
  Checkpoint: 500 MB
  Subtotal: ~2.8 GB

Total: ~8.3 GB (with some overlap possible)
```

---

## 6. DECISION TREE: Which Model to Use When?

```
Scenario 1: Know learner proficiency level
  → Use per-CEFR model for that level
  → Maximum specialization

Scenario 2: Don't know learner proficiency
  → Use general model
  → Works for all levels (compromised)

Scenario 3: Want best possible performance
  → If per-CEFR models win: Use specialization
  → If general model wins: Use single model
  → If tied: Use general for simplicity

Scenario 4: Limited deployment resources
  → Use general model
  → 1 model vs 6 models
  → 2.8 GB vs 5.5 GB storage

Scenario 5: Continuous learner tracking
  → Start with general model
  → Fine-tune as proficiency level detected
  → Dynamic transfer learning approach
```

---

## 7. RESOURCE-CONSTRAINED ALTERNATIVES

If cannot train all 6 models:

### Option A: Train Per-CEFR Only (Skip General)
- **Pros**: Get specialization hypothesis
- **Cons**: No baseline, can't compare
- **Recommendation**: SKIP this option

### Option B: Train General + Top 2 Per-CEFR
- Train General (all samples) - baseline
- Train B2 (mid-range, 40K samples) - validation
- Train A1 (largest, 341K samples) - comparison
- **Cost**: ~6 hours
- **Benefit**: Quick validation of hypothesis

### Option C: Train General + C1 + A1
- Train General (baseline)
- Train C1 (smallest, most vulnerable)
- Train A1 (largest, most different)
- **Cost**: ~8 hours
- **Benefit**: Extreme cases show if specialization matters

### Option D: Train General Only
- Single baseline model
- Comparison against per-CEFR impossible
- Not recommended for research

---

## 8. CRITICAL SUCCESS METRICS

### Per-CEFR Models are "Better" if:
1. **Metric 1: Specialization Gain** ≥ 25%
   - C1 model at C1: 25% better than general model at C1

2. **Metric 2: Level Specificity** ≥ 3x
   - Model 3x better on its level than other levels

3. **Metric 3: Small Dataset Success**
   - C1 model (10K samples) converges to acceptable perplexity
   - Not worse than general model

### General Model is "Better" if:
1. **Metric 1: Simplicity** - 1 model vs 6
   - Deployment advantage

2. **Metric 2: Robustness** - PPL within 10% of best per-CEFR
   - Single model competitive

3. **Metric 3: Mixed-Level** - Best on combined test set
   - Realistic scenario advantage

---

## 9. TIMELINE

```
Execution Order:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1: Extract & Prepare Data
  ✅ General model data: EXTRACTED (185 MB TXT files)
  ⏳ General model examples: CREATING (9.2M examples)
  ⏳ Per-CEFR data: Ready to extract

Phase 2: Training (Estimate)
  Day 1: General model (5 hours) + C1 model (2 hours)
  Day 2: B2 model (4 hours) + B1 model (6 hours)
  Day 3: A2 model (8 hours)
  Day 4: A1 model (10 hours)

Phase 3: Evaluation (1 day)
  - Perplexity on all test sets
  - Cross-level evaluation
  - Comparison analysis

Phase 4: Analysis (1 day)
  - Plot results
  - Statistical significance
  - Recommendations
```

---

## 10. EXPECTED PAPER/REPORT STRUCTURE

Based on general vs per-CEFR comparison:

```
Title: "Specialized vs. General Language Models for L2 Learner Text:
        An ILM Study on EFCAMDAT"

Sections:
1. Introduction
   - Problem: How to model heterogeneous learner language?
   - Hypothesis: Specialization improves performance

2. Methods
   - EFCAMDAT dataset (723K samples)
   - ILM training approach
   - 6 models: 5 per-CEFR + 1 general

3. Results
   - Perplexity comparison table
   - Graphs: Specialization gain by level
   - Cross-level analysis
   - Transfer learning results

4. Discussion
   - Does specialization help? By how much?
   - Cost-benefit analysis
   - Practical deployment recommendations

5. Conclusion
   - Recommendations for practitioners
   - Future directions (multilingual, other task types)
```

---

## 11. NEXT STEPS

1. ⏳ Complete general model training example creation (in progress)
2. ⏳ Complete validation examples for general model
3. Train general model (60,000 steps, ~5 hours)
4. Train per-CEFR models (C1→B2→B1→A2→A1)
5. Evaluate all 6 models
6. Compare results & answer: **Does specialization help?**

---

**This general model is the CRITICAL BASELINE for answering whether the per-CEFR specialization strategy is actually beneficial.**

**Status**: General model data extraction ✅ COMPLETE
**Next**: Masked examples creation ⏳ IN PROGRESS
