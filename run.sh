#!/bin/bash
# =============================================================================
# Run Script: Text-to-SMILES Fine-tuning with OFT
# =============================================================================
# Usage:
#   1. Prepare data:   bash run.sh prepare
#   2. Train model:    bash run.sh train
#   3. Evaluate OFT:   bash run.sh eval_oft
#   4. Evaluate base:  bash run.sh eval_base
#   5. Run all:        bash run.sh all
# =============================================================================

set -e

MODEL_NAME="Qwen/Qwen3.5-0.8B"
OUTPUT_DIR="output/qwen35-oft-smiles"
TRAIN_FILE="data/train.jsonl"
TEST_FILE="data/test.jsonl"
MAX_EVAL_SAMPLES=1000  # Match the 1,000-sample evaluation reported in report.tex
OFT_RESULTS_FILE="results/oft_results_qwen35_1k.json"
BASE_RESULTS_FILE="results/base_results_qwen35_1k.json"

prepare_data() {
    echo "=== Step 1: Preparing data ==="
    python3 prepare_data.py
    echo "Done. Train/test JSONL files saved to data/"
}

train_model() {
    echo "=== Step 2: Training with OFT ==="
    python3 train.py \
        --model_name ${MODEL_NAME} \
        --train_file ${TRAIN_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --logging_steps 50 \
        --save_steps 500 \
        --bf16 \
        --gradient_checkpointing
    echo "Done. Model saved to ${OUTPUT_DIR}"
}

eval_oft() {
    echo "=== Step 3: Evaluating fine-tuned model ==="
    python3 evaluate.py \
        --model_name ${OUTPUT_DIR} \
        --base_model_name ${MODEL_NAME} \
        --test_file ${TEST_FILE} \
        --output_file ${OFT_RESULTS_FILE} \
        --max_samples ${MAX_EVAL_SAMPLES}
}

eval_base() {
    echo "=== Step 4: Evaluating base model (zero-shot) ==="
    python3 evaluate.py \
        --model_name ${MODEL_NAME} \
        --test_file ${TEST_FILE} \
        --output_file ${BASE_RESULTS_FILE} \
        --max_samples ${MAX_EVAL_SAMPLES}
}

case "$1" in
    prepare) prepare_data ;;
    train)   train_model ;;
    eval_oft)  eval_oft ;;
    eval_base) eval_base ;;
    all)
        prepare_data
        train_model
        eval_oft
        eval_base
        ;;
    *)
        echo "Usage: bash run.sh {prepare|train|eval_oft|eval_base|all}"
        exit 1
        ;;
esac
