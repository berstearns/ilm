# Ready to Train General Model - Execute When Files Appear

**Status**: Waiting for masked examples to complete (~10-20 minutes)

**Monitoring**: Automatic script running - will alert when ready

---

## ✅ WHAT'S READY

- ✅ Data extracted: `data/efcamdat_all/{train,valid,test}.txt` (232 MB)
- ⏳ Masked examples creating: `data/char_masks/efcamdat_all/{train,valid}.pkl`
- ✅ Training config prepared
- ✅ All directories created

---

## 🚀 TRAIN GENERAL MODEL (Execute When .pkl Files Appear)

Once you see this confirmation:
```
[COMPLETE] HH:MM:SS - Both files ready!
  train.pkl: ~2.3G
  valid.pkl: ~290M
```

Run this command:

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

**Expected Runtime**: 14-16 GPU hours

**Output Location**: `experiments/efcamdat_all_ilm/pytorch_model.bin` (~500 MB)

---

## 📊 THEN: Extract Per-CEFR Data (Parallel with Training)

While the general model trains, prepare per-CEFR data:

```bash
# C1 (smallest, fastest)
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_C1 \
  --cefr_level C1 \
  --seed 0

# B2
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_B2 \
  --cefr_level B2 \
  --seed 0

# B1
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_B1 \
  --cefr_level B1 \
  --seed 0

# A2
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_A2 \
  --cefr_level A2 \
  --seed 0

# A1
~/.pyenv/versions/3.9.25/bin/python scripts/csv_to_txt_efcamdat.py \
  --csv_path /home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/norm-EFCAMDAT-ALL-CONCAT.csv \
  --output_dir data/efcamdat_A1 \
  --cefr_level A1 \
  --seed 0
```

---

## 📝 COMPLETE PIPELINE IN EFCAMDAT_TRAINING_RUN.MD

See `efcamdat_training_run.md` for:
- All 6 model training commands
- Evaluation procedures
- Analysis templates
- Expected outputs

---

**Last Updated**: 2026-01-29 15:15 UTC
**Monitoring**: Active - will update when files appear
