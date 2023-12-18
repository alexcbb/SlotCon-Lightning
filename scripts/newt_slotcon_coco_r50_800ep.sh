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


module purge

source ~/.bashrc

conda activate mttoc

export HYDRA_FULL_ERROR=1

data_dir="./data/COCO/"

srun python main_lightning_train.py \
    --dataset COCO \
    --data-dir ${data_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 1024 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --num-workers 8 \
    --gpus 2 