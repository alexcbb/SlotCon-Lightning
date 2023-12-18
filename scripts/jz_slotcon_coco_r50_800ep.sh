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

srun main_lightning_train.py --dataset COCO --data-dir ${data_dir} --arch resnet50 --epochs 800 --fp16 --num-workers 8


