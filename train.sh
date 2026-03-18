#!/bin/sh

# Multi-GPU training launcher for StableVSR (requires CUDA).
# Update the paths below before running.

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/my_exp_name'
GPUS="5 6 7 8"

# Dataset and validation paths — set these to your local filesystem
DATASET_CONFIG="dataset/config_reds.yaml"
VALIDATION_FRAMES="/path/to/REDS/train/bicubic/020/00000000.png;/path/to/REDS/train/bicubic/020/00000001.png;/path/to/REDS/train/bicubic/020/00000002.png;/path/to/REDS/train/bicubic/020/00000003.png;/path/to/REDS/train/bicubic/020/00000004.png;/path/to/REDS/train/bicubic/020/00000005.png;/path/to/REDS/train/bicubic/020/00000006.png;/path/to/REDS/train/bicubic/020/00000007.png;/path/to/REDS/train/bicubic/020/00000008.png;/path/to/REDS/train/bicubic/020/00000009.png"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR

# Calculate the number of GPUs (i.e., the number of processes)
NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29501 train.py \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path="$DATASET_CONFIG" \
 --learning_rate=5e-5 \
 --validation_steps=1000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=20000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "$VALIDATION_FRAMES" \
 --validation_prompt ""
