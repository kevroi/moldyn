#!/bin/bash
#SBATCH --job-name=PPO
#SBATCH --account=rrg-whitem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-2:59
#SBATCH --mail-type=ALL


echo $MOL, $LR

module load python/3.9
source venv/bin/activate
wandb login

python $me/predicting-MolDyn/train_model.py --molecule=$MOL --lr=$LR --use_wandb