#!/bin/bash
#SBATCH --job-name=tcav_recalib
#SBATCH --output=logs/tcav_%j.out
#SBATCH --error=logs/tcav_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ============================================================================
# TCAV-Based Recalibration - SLURM Batch Script
# ============================================================================
# 
# This script runs Experiment 3 with multiple imbalance ratios to study
# how TCAV-based recalibration performs under different imbalance conditions.
#
# Usage:
#   sbatch gpu.sh
#
# Or submit with custom parameters:
#   sbatch --export=IMBALANCE_RATIO=0.1,IMBALANCE_CLASS=airplanes gpu.sh
# ============================================================================

# Create logs directory
mkdir -p logs

# Load required modules (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate virtual environment (adjust path as needed)
# source ~/venvs/tcav/bin/activate

# Set environment variables for reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Default configuration
MODEL_NAME="${MODEL_NAME:-custom_cnn}"
DATASET_PATH="${DATASET_PATH:-./data}"
CONCEPT_PATH="${CONCEPT_PATH:-./concepts}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
RECALIB_EPOCHS="${RECALIB_EPOCHS:-20}"
LAMBDA_ALIGN="${LAMBDA_ALIGN:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

# Caltech-101 Vehicle Classes Configuration
# Available classes: airplanes, Motorbikes, car_side, ferry, helicopter, schooner
CLASS_CONCEPT_MAP="${CLASS_CONCEPT_MAP:-airplanes:subject,Motorbikes:subject,car_side:subject,ferry:subject,helicopter:subject}"
IMBALANCE_CLASS="${IMBALANCE_CLASS:-airplanes}"

echo "=============================================="
echo "TCAV-Based Recalibration Experiment"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="
echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Classes: ${CLASS_CONCEPT_MAP}"
echo "  Imbalance Class: ${IMBALANCE_CLASS}"
echo "  Pretrain Epochs: ${PRETRAIN_EPOCHS}"
echo "  Recalib Epochs: ${RECALIB_EPOCHS}"
echo "  Lambda Align: ${LAMBDA_ALIGN}"
echo "=============================================="

# ============================================================================
# Single Imbalance Ratio Run
# ============================================================================
# Uncomment this section to run a single experiment

# IMBALANCE_RATIO="${IMBALANCE_RATIO:-0.1}"
# 
# python main_experiment.py \
#     --experiment 3 \
#     --model_name ${MODEL_NAME} \
#     --dataset_path ${DATASET_PATH} \
#     --concept_path ${CONCEPT_PATH} \
#     --class_concept_map "${CLASS_CONCEPT_MAP}" \
#     --imbalance_class ${IMBALANCE_CLASS} \
#     --imbalance_ratio ${IMBALANCE_RATIO} \
#     --pretrain_epochs ${PRETRAIN_EPOCHS} \
#     --recalib_epochs ${RECALIB_EPOCHS} \
#     --lambda_align ${LAMBDA_ALIGN} \
#     --batch_size ${BATCH_SIZE} \
#     --seed ${SEED}

# ============================================================================
# Multiple Imbalance Ratios Run
# ============================================================================
# Run experiments with different imbalance ratios to study the effect

IMBALANCE_RATIOS=(0.05 0.10 0.15 0.20 0.25 0.50)

for ratio in "${IMBALANCE_RATIOS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running with imbalance ratio: ${ratio}"
    echo "=============================================="
    
    python main_experiment.py \
        --experiment 3 \
        --model_name ${MODEL_NAME} \
        --dataset_path ${DATASET_PATH} \
        --concept_path ${CONCEPT_PATH} \
        --class_concept_map "${CLASS_CONCEPT_MAP}" \
        --imbalance_class ${IMBALANCE_CLASS} \
        --imbalance_ratio ${ratio} \
        --pretrain_epochs ${PRETRAIN_EPOCHS} \
        --recalib_epochs ${RECALIB_EPOCHS} \
        --lambda_align ${LAMBDA_ALIGN} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED}
    
    echo "Completed ratio ${ratio}"
    echo ""
done

echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: ./results/"
echo "=============================================="

# ============================================================================
# Alternative: Run with Different Models
# ============================================================================
# Uncomment to compare different model architectures

# MODELS=(custom_cnn custom_cnn_small vgg16 resnet18)
# IMBALANCE_RATIO=0.1
# 
# for model in "${MODELS[@]}"; do
#     echo "Running with model: ${model}"
#     
#     python main_experiment.py \
#         --experiment 3 \
#         --model_name ${model} \
#         --dataset_path ${DATASET_PATH} \
#         --concept_path ${CONCEPT_PATH} \
#         --class_concept_map "${CLASS_CONCEPT_MAP}" \
#         --imbalance_class ${IMBALANCE_CLASS} \
#         --imbalance_ratio ${IMBALANCE_RATIO} \
#         --pretrain_epochs ${PRETRAIN_EPOCHS} \
#         --recalib_epochs ${RECALIB_EPOCHS} \
#         --seed ${SEED}
# done

# ============================================================================
# Alternative: Run with Pretrained Models
# ============================================================================
# Uncomment to use ImageNet pretrained weights

# python main_experiment.py \
#     --experiment 3 \
#     --model_name vgg16 \
#     --pretrained \
#     --dataset_path ${DATASET_PATH} \
#     --concept_path ${CONCEPT_PATH} \
#     --class_concept_map "${CLASS_CONCEPT_MAP}" \
#     --imbalance_class ${IMBALANCE_CLASS} \
#     --imbalance_ratio 0.1 \
#     --pretrain_epochs 10 \
#     --recalib_epochs 5 \
#     --seed ${SEED}
