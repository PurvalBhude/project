#!/bin/bash
#SBATCH --partition=Brain3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --output=log/%x/%j/logs.out
#SBATCH --error=log/%x/%j/errors.err

export WANDB_DIR=/Brain/private/username/wandb/
export WANDB_MODE=offline

# do every part that need to be done such as activating python environment and running the script
source venv/bin/activate
srun python3 main.py