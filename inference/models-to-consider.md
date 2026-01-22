# Modern Infilling Models to Consider (2024-2025)

A comprehensive guide to SOTA models that work out of the box for infilling evaluation.

## MLM Models (Fill-Mask Pipeline)

### Tier 1: SOTA Recommended

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **ModernBERT** | `answerdotai/ModernBERT-base` | 149M | Best choice for 2024/2025. 2-4x faster than DeBERTa, 8K context, SOTA on GLUE. Uses `[MASK]` |
| **ModernBERT Large** | `answerdotai/ModernBERT-large` | 395M | Larger version, even better accuracy |
| **RoBERTa** | `roberta-base`, `roberta-large` | 125M/355M | Still solid baseline, well-tested |
| **XLM-RoBERTa** | `FacebookAI/xlm-roberta-base` | 270M | Multilingual (100 languages). Uses `<mask>` |

### Tier 2: Efficient/Distilled

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **DistilRoBERTa** | `distilbert/distilroberta-base` | 82M | 2x faster than RoBERTa, 92.7% perf |
| **DistilBERT** | `distilbert/distilbert-base-uncased` | 66M | 60% faster than BERT, 97% perf |
| **ALBERT** | `albert-base-v2`, `albert-large-v2` | 12M/18M | Parameter-efficient, good for low-memory |

### Avoid for Fill-Mask

| Model | Issue |
|-------|-------|
| **DeBERTa v3** | [Known bug](https://github.com/huggingface/transformers/issues/22790) - produces nonsense for fill-mask |
| **ELECTRA** | Not trained for MLM (uses replaced token detection) |

---

## Seq2Seq Models (T5-style Infilling)

### Tier 1: SOTA Recommended

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **Flan-T5** | `google/flan-t5-base`, `google/flan-t5-large`, `google/flan-t5-xl` | 250M/780M/3B | Instruction-tuned T5, works zero-shot |
| **Flan-UL2** | `google/flan-ul2` | 20B | SOTA on 50+ NLP tasks, 2K context, no mode tokens needed |
| **mT5** | `google/mt5-base`, `google/mt5-large` | 580M/1.2B | Multilingual T5 (101 languages) |

### Tier 2: Specialized

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **CodeT5+** | `Salesforce/codet5p-220m`, `Salesforce/codet5p-770m` | 220M/770M | Code understanding/generation |
| **mBART-50** | `facebook/mbart-large-50` | 611M | Multilingual, primarily for translation |

### Advanced: GLM (Autoregressive Blank Infilling)

| Model | HuggingFace ID | Params | Notes |
|-------|----------------|--------|-------|
| **GLM-10B** | `THUDM/glm-10b` | 10B | Native blank infilling, outperforms BERT/T5 on NLU |
| **GLM-4-9B** | `THUDM/glm-4-9b-chat-hf` | 9B | Latest, 128K context, multi-language |

---

## Usage Examples

### MLM Models (fill-mask)

```bash
python ilm_eval.py -i data.csv --models \
  mlm:answerdotai/ModernBERT-base \
  mlm:roberta-base \
  mlm:distilbert/distilroberta-base \
  mlm:FacebookAI/xlm-roberta-base \
  mlm:albert-base-v2
```

### Seq2Seq Models (T5-style)

```bash
python ilm_eval.py -i data.csv --models \
  t5:google/flan-t5-base \
  t5:google/flan-t5-large \
  t5:google/mt5-base
```

### Full Comparison (mixed)

```bash
python ilm_eval.py -i data.csv --models \
  ilm:../models/sto_ilm \
  mlm:answerdotai/ModernBERT-base \
  mlm:roberta-large \
  mlm:distilbert/distilroberta-base \
  t5:google/flan-t5-base \
  t5:google/flan-t5-large \
  --n-masks 5 --print-every 100 -o results.json
```

---

## Key Takeaways

1. **ModernBERT** is the new gold standard for MLM - faster and more accurate than everything before it
2. **Flan-T5** is the best drop-in T5 replacement - instruction-tuned, works zero-shot
3. **DeBERTa v3 has a known bug** for fill-mask tasks despite being excellent for classification
4. For **multilingual**: Use XLM-RoBERTa (MLM) or mT5 (seq2seq)
5. For **efficiency**: DistilRoBERTa gives 92.7% of RoBERTa performance at 2x speed

---

## References

- [ModernBERT Blog](https://huggingface.co/blog/modernbert)
- [ModernBERT vs DeBERTaV3 Paper](https://arxiv.org/html/2504.08716v1)
- [HuggingFace Fill-Mask Models](https://huggingface.co/models?pipeline_tag=fill-mask)
- [Flan-UL2 Model Card](https://huggingface.co/google/flan-ul2)
- [DeBERTa Fill-Mask Issue](https://github.com/huggingface/transformers/issues/22790)
- [XLM-RoBERTa Documentation](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
- [Yi Tay's Blog on Encoders](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising)
