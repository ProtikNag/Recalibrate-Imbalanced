#!/bin/sh
#SBATCH --job-name=TCAV
#SBATCH -N 1         			## Compute None (Number of computers)
#SBATCH -n 24 	     			## CPU Cores 
#SBATCH --gres=gpu:1 			## Run on 2 GPUs
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p dgx_aic

hostname
date

export CUDA_VISIBLE_DEVICES=0

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/Recalibrate-Imbalanced/

# Experiment 1: Target class only recalibration
python main_experiment.py \
    --experiment 1 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.05 \
    --pretrain_epochs 50 \
    --recalib_epochs 20

# Experiment 2: Full dataset with selective alignment
python main_experiment.py \
    --experiment 2 \
    --model_name custom_cnn \
    --layer features.17 \
    --target_class zebra \
    --concept stripes \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --imbalance_class zebra \
    --imbalance_ratio 0.05 \
    --pretrain_epochs 50 \
    --recalib_epochs 20

# Experiment 3: Joint multi-class optimization with automatic layer selection
python main_experiment.py \
    --experiment 3 \
    --model_name custom_cnn \
    --dataset_path ./dataset \
    --concept_path ./concept \
    --class_concept_map "zebra:stripes,horse:horse_skin,deer:antlers" \
    --imbalance_class zebra \
    --imbalance_ratio 0.05 \
    --pretrain_epochs 50 \
    --recalib_epochs 20
