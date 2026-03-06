#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-llama3.1:8b}"
BASE_URL="${BASE_URL:-http://localhost:11434}"
SEED=42

echo "================================================"
echo "Prompt Ablation Evaluation Pipeline"
echo "================================================"
echo "Model:    $MODEL"
echo "Endpoint: $BASE_URL"
echo "Seed:     $SEED"
echo ""

echo "[1/4] Preparing data..."
python ablation/prepare_data.py --seed $SEED

echo ""
echo "[2/4] Running prompt optimization ablation..."
python ablation/optimize_prompt.py \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --max-examples 50 \
    --seed $SEED

echo ""
echo "[3/4] Running improved inference with comparison..."
python ablation/infer.py \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --max-examples 50 \
    --seed $SEED

echo ""
echo "[4/4] Running with self-consistency..."
python ablation/infer.py \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --max-examples 30 \
    --self-consistency \
    --k 5 \
    --seed $SEED

echo ""
echo "================================================"
echo "Pipeline complete. Results in ablation/results/"
echo "================================================"
