#!/bin/bash
#SBATCH --job-name=baseline  
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --ntasks-per-node=1        # Use 1 task (process) per node
#SBATCH -c 32         
#SBATCH --mem=32G
##SBATCH --gpus=1                   # Request 32 GB of memory 
#SBATCH --gres=gpu:a100:1          # Request 1 Nvidia A100 GPU
#SBATCH --time=0-00:10:00


module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
conda deactivate
source $STORE/mytorchdist/bin/deactivate
source $STORE/mytorchdist/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

which python
srun python baseline.py