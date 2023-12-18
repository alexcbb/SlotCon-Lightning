#!/bin/bash
#SBATCH --job-name=slotcon
#SBATCH --output=logs/slotcon/slotcon.%j.out
#SBATCH --error=logs/slotcon/slotcon.%j.err

#SBATCH -A uli@v100
#SBATCH --partition=gpu_p2s
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8 #number of MPI tasks per node (=number of GPUs per node)
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH -t 50:00:00
#SBATCH --mail-user=alexandre.chapin@ec-lyon.fr
#SBATCH --mail-typ=FAIL
#SBATCH --qos=qos_gpu-t4

echo ${SLURM_NODELIST}

module purge
data_dir="./data/COCO/"

#module load cpuarch/amd # To be compatible with a100 nodes
module load pytorch-gpu/py3/2.0.0

export HYDRA_FULL_ERROR=1


srun main_lightning_train.py \
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
    --batch-size 768 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --num-workers 8


