#!/bin/bash
#SBATCH --job-name=slotcon
#SBATCH --output=logs/slotcon/slotcon.%j.out
#SBATCH --error=logs/slotcon/slotcon.%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL

echo ${SLURM_NODELIST}

set -e
set -x

data_dir="./datasets/coco/"
output_dir="./output/slotcon_coco_r50_800ep"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 12348 --nproc_per_node=8 \
    main_pretrain.py \
    --dataset COCO \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 512 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 50 \
    --auto-resume \
    --num-workers 8
