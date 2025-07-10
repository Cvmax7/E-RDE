#!/bin/bash

# Enhanced RDE Training Script for RSTPReID Dataset
# This script runs the enhanced RDE model with all improvements on RSTPReID dataset

echo "Starting Enhanced RDE training on RSTPReID dataset..."

python train.py \
    --name enhanced_rde_rstpreid \
    --img_size 384 \
    --stride_size 128 \
    --batch_size 64 \
    --num_epoch 80 \
    --lr 1e-4 \
    --lr_update 40 \
    --eval_period 5 \
    --log_period 50 \
    --dataset_name "RSTPReid" \
    --pretrain_choice "ViT-B/16" \
    --loss_names "TAL" \
    --select_ratio 0.5 \
    --tau 0.02 \
    --margin 0.2 \
    --noisy_rate 0.2 \
    --noisy_file "./noiseindex/RSTPReid_0.2.npy" \
    --root_dir "./datasets" \
    --use_enhanced_rde \
    --num_evidence_views 4 \
    --uncertainty_threshold 0.5 \
    --confidence_threshold 0.7 \
    --cost_hidden_dim 128 \
    --ot_reg 0.1 \
    --ot_max_iter 100 \
    --rematch_temperature 0.07 \
    --ccl_temperature 0.07 \
    --push_phase_epochs 15 \
    --transition_epochs 5 \
    --evidence_reg 0.1 \
    --warmup_epochs 5 \
    --uncertainty_epochs 10 \
    --progressive_epochs 65 \
    --output_dir "./enhanced_rde_rstpreid_results" \
    --img_aug \
    --txt_aug

echo "Enhanced RDE training on RSTPReID completed!"
echo "Results saved to: ./enhanced_rde_rstpreid_results" 