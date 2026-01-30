#!/bin/bash

##############################################################################
# TEST SCRIPT: Minimal ILM Model Evaluation with ilm_eval.py
#
# This script demonstrates:
# 1. Creating minimal test data (2-3 samples)
# 2. Running ilm_eval.py with trained ILM model
# 3. Verifying output format and metrics
#
# Usage: bash test_ilm_eval_example.sh
##############################################################################

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  ILM Model Evaluation Test - Comprehensive End-to-End Demo        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
PROJECT_ROOT="/home/b/p/research-sketches/ilms"
PYTHON_BIN="~/.pyenv/versions/3.9.25/bin/python"
TEST_DATA_CSV="/tmp/ilm_test_data.csv"
TEST_OUTPUT_JSON="/tmp/ilm_test_results.json"
MODEL_PATH="experiments/efcamdat_test_sample"

echo "📋 Step 1: Creating test data with 3 samples (2-3 CEFR levels)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create minimal test CSV with CEFR levels
cat > "$TEST_DATA_CSV" << 'EOF'
text,cefr
"I like to eat apples and bananas with my friends.",A1
"The weather is very nice today and I enjoy walking in the park.",A2
"Education plays a crucial role in the development of individuals and society.",B1
EOF

echo "✓ Test data created: $TEST_DATA_CSV"
echo ""
echo "Content:"
cat "$TEST_DATA_CSV"
echo ""

echo ""
echo "📁 Step 2: Verifying model files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT" || exit 1

echo "Checking model directory: $MODEL_PATH"
echo ""

if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model directory exists"
    echo ""
    echo "📦 Files in model directory:"
    ls -lh "$MODEL_PATH"/ | grep -E "pytorch_model|additional_ids|config|tokenizer|vocab|merges"
    echo ""

    # Check critical file
    if [ -f "$MODEL_PATH/additional_ids_to_tokens.pkl" ]; then
        echo "✓ CRITICAL: additional_ids_to_tokens.pkl found"
    else
        echo "⚠ WARNING: additional_ids_to_tokens.pkl NOT found"
        echo "  This file is required for ILM evaluation"
    fi
else
    echo "✗ ERROR: Model directory not found at $MODEL_PATH"
    exit 1
fi

echo ""

echo ""
echo "🚀 Step 3: Running ilm_eval.py with minimal configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Command:"
echo "  ~/.pyenv/versions/3.9.25/bin/python inference/ilm_eval.py \\"
echo "    -i $TEST_DATA_CSV \\"
echo "    --models ilm:$MODEL_PATH \\"
echo "    --limit 3 \\"
echo "    --n-masks 1 \\"
echo "    --samples-per-text 5 \\"
echo "    --print-every 1 \\"
echo "    --seed 42 \\"
echo "    -o $TEST_OUTPUT_JSON"
echo ""

# Run evaluation (using ~/.pyenv/versions... won't work in subshell, so use full path)
eval "$PYTHON_BIN inference/ilm_eval.py \
  -i $TEST_DATA_CSV \
  --models ilm:$MODEL_PATH \
  --limit 3 \
  --n-masks 1 \
  --samples-per-text 5 \
  --print-every 1 \
  --seed 42 \
  -o $TEST_OUTPUT_JSON"

echo ""
echo "✓ Evaluation completed!"
echo ""

echo ""
echo "📊 Step 4: Examining results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "$TEST_OUTPUT_JSON" ]; then
    echo "✓ Results file created: $TEST_OUTPUT_JSON"
    echo ""
    echo "📋 Results (formatted JSON):"
    echo "─────────────────────────────────────────────────────────────"
    eval "$PYTHON_BIN" -m json.tool < "$TEST_OUTPUT_JSON"
    echo "─────────────────────────────────────────────────────────────"
else
    echo "✗ ERROR: Results file not created"
    exit 1
fi

echo ""

echo ""
echo "📈 Step 5: Key metrics extraction"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Extract and display key metrics using Python
eval "$PYTHON_BIN" << 'PYTHON_SCRIPT'
import json
import sys

try:
    with open('/tmp/ilm_test_results.json', 'r') as f:
        results = json.load(f)

    print("✓ Successfully parsed results JSON")
    print()

    # Overall metrics
    if 'results' in results and 'overall' in results['results']:
        overall = results['results']['overall']

        for model_name, metrics in overall.items():
            print(f"Model: {model_name}")
            print("├─ Accuracy (Top-1):   {:.1%}".format(metrics.get('top_1_accuracy', 0)))
            print("├─ Unigram Recall:     {:.1%}".format(metrics.get('unigram_recall', 0)))
            print("├─ Unigram F1:         {:.1%}".format(metrics.get('unigram_f1', 0)))
            print("├─ Bigram Recall:      {:.1%}".format(metrics.get('bigram_recall', 0)))
            print("├─ Bigram F1:          {:.1%}".format(metrics.get('bigram_f1', 0)))
            print("└─ Samples Evaluated:  {}".format(metrics.get('samples', 0)))
            print()

    # By CEFR breakdown
    if 'results' in results and 'by_cefr' in results['results']:
        by_cefr = results['results']['by_cefr']

        print("Breakdown by CEFR Level:")
        print("─────────────────────────")
        for cefr_level in sorted(by_cefr.keys()):
            cefr_data = by_cefr[cefr_level]
            if isinstance(cefr_data, dict) and 'samples' in cefr_data:
                samples = cefr_data.get('samples', 0)
                print(f"  {cefr_level}: {samples} samples")

                # Get model metrics if available
                for model_name, metrics in cefr_data.items():
                    if model_name != 'samples' and isinstance(metrics, dict):
                        accuracy = metrics.get('top_1_accuracy', 0)
                        print(f"    └─ {model_name}: {accuracy:.1%} accuracy")
        print()

    print("✓ Metrics successfully extracted")

except Exception as e:
    print(f"✗ Error parsing results: {e}")
    sys.exit(1)

PYTHON_SCRIPT

echo ""

echo ""
echo "✅ TEST COMPLETE"
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Success! ILM model evaluation works end-to-end                  ║"
echo "║                                                                  ║"
echo "║  Next Steps:                                                     ║"
echo "║  1. Evaluate on full dataset: --limit 100                        ║"
echo "║  2. Run multi-model comparison (general + specialized)           ║"
echo "║  3. Analyze CEFR-level breakdowns                                ║"
echo "║                                                                  ║"
echo "║  For details, see: ILM_MODEL_EVALUATION_GUIDE.md                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📁 Output files:"
echo "   Test data:   $TEST_DATA_CSV"
echo "   Results:     $TEST_OUTPUT_JSON"
echo ""
echo "To view results again: python -m json.tool < $TEST_OUTPUT_JSON"
echo ""

